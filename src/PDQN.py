# agents/pdqn_custom.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable  # For older PyTorch versions
import os
from agent import Agent  # Adjusted import
from memory import Memory  # Adjusted import
from network_utils import soft_update_target_network, hard_update_target_network  # Adjusted import
from noise import OrnsteinUhlenbeckActionNoise  # Adjusted import


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size  # This is total size of actual parameters
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        # Input to QActor is state + concatenated actual parameters
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))  # Outputs Q-value for each discrete action

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None,
                 device='cpu'):  # Added device
        super(ParamActor, self).__init__()
        self.device = device  # Store device
        self.state_size = state_size
        self.action_size = action_size  # Not directly used for output size here
        self.action_parameter_size = action_parameter_size  # This IS the output size: sum of dims of actual parameters
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)

        # Passthrough layer can be problematic if state features don't directly map to parameters
        # For simplicity, I'm commenting it out. If you need it, ensure dimensions match.
        # self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type " + str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        # nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        # nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        # self.action_parameters_passthrough_layer.requires_grad = False
        # self.action_parameters_passthrough_layer.weight.requires_grad = False
        # self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        # action_params += self.action_parameters_passthrough_layer(state) # If using passthrough


        return action_params


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """
    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,  # gym.spaces.Tuple: (Discrete(N), Box(P0), Box(P1)...)
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=1000,
                 batch_size=128,
                 gamma=0.99,
                 tau_actor=0.01,
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=False,  # Often True for DDPG variants
                 zero_index_gradients=True,#False,
                 indexed=False,  # For Q_loss calculation type
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device='cpu',
                 seed=None,start_epsilon=0):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)

        self.num_actions = self.action_space.spaces[0].n
        # action_parameter_sizes: array of param dimensions for each discrete action
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i + 1].shape[0] for i in range(self.num_actions)])
        # action_parameter_size: total dimension of all actual parameters (sum of non-zero P_i)
        self.action_parameter_size = int(self.action_parameter_sizes.sum())

        # Parameter limits - derived from concatenated actual parameters
        # Correctly handles shape=(0,) for actions without params due to np.concatenate behavior
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i + 1].high for i in range(self.num_actions)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i + 1].low for i in range(self.num_actions)]).ravel()

        # Ensure they are not empty if all actions had 0 params (edge case, not for this problem)
        if self.action_parameter_size > 0:
            self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
            self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(self.device)
            self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(self.device)
            self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(self.device)
        else:  # No continuous parameters at all
            self.action_parameter_range_numpy = np.array([])
            self.action_parameter_max = torch.tensor([], dtype=torch.float32).to(self.device)
            self.action_parameter_min = torch.tensor([], dtype=torch.float32).to(self.device)
            self.action_parameter_range = torch.tensor([], dtype=torch.float32).to(self.device)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert not (weighted and average and random_weighted and indexed), "Choose one or none of Q_loss types"

        # Offsets for slicing the concatenated parameter vector (size = self.action_parameter_size)
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)
        # For [1,1,0,0], cumsum is [1,2,2,2], offsets are [0,1,2,2,2]
        # This means actual parameters are stored contiguously.
        # The ParamActor outputs a vector of size sum(action_parameter_sizes).
        # The QActor (MultiPassQActor) receives this concatenated vector.

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = start_epsilon #0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise and self.action_parameter_size > 0
        if self.use_ornstein_noise:
            self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0.,
                                                      theta=0.15, sigma=0.0001)
        else:
            self.noise = None

        # Replay memory stores: state, (discrete_action_idx, all_actual_params_vector), reward, next_state, (next_discrete_idx, next_all_actual_params), terminal
        # Shape of all_actual_params_vector is (self.action_parameter_size,)
        # So action_shape for memory is (1 + self.action_parameter_size,)
        self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1 + self.action_parameter_size,),
                                    next_actions=True, seed=seed)  # Store next_actions

        # actor_kwargs and actor_param_kwargs might have 'device', remove if PDQNAgent passes it
        actor_kwargs_clean = {k: v for k, v in actor_kwargs.items() if k != 'device'}
        actor_param_kwargs_clean = {k: v for k, v in actor_param_kwargs.items() if k != 'device'}

        # QActor takes total size of actual parameters
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                 **actor_kwargs_clean).to(self.device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size,
                                        **actor_kwargs_clean).to(self.device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        # ParamActor outputs vector of total actual parameters
        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                             self.action_parameter_size, device=self.device,output_layer_init_std=0.1,
                                             **actor_param_kwargs_clean).to(self.device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions,
                                                    self.action_parameter_size, device=self.device,output_layer_init_std=0.1,
                                                    **actor_param_kwargs_clean).to(self.device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Device: {}\n".format(self.device) + \
                "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Num Actions: {}\n".format(self.num_actions) + \
                "Action Parameter Sizes: {}\n".format(self.action_parameter_sizes) + \
                "Total Action Parameter Size: {}\n".format(self.action_parameter_size) + \
                "Action Param Max: {}\n".format(self.action_parameter_max_numpy) + \
                "Action Param Min: {}\n".format(self.action_parameter_min_numpy) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        # This was for a specific ParamActor design, may need adjustment if passthrough is used.
        # For now, assuming passthrough is not the primary mechanism.
        if hasattr(self.actor_param, 'action_parameters_passthrough_layer'):
            passthrough_layer = self.actor_param.action_parameters_passthrough_layer
            print(initial_weights.shape)
            print(passthrough_layer.weight.data.size())
            assert initial_weights.shape == passthrough_layer.weight.data.size()
            passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
            if initial_bias is not None:
                print(initial_bias.shape)
                print(passthrough_layer.bias.data.size())
                assert initial_bias.shape == passthrough_layer.bias.data.size()
                passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
            passthrough_layer.requires_grad = False
            passthrough_layer.weight.requires_grad = False
            passthrough_layer.bias.requires_grad = False
            hard_update_target_network(self.actor_param, self.actor_param_target)
        else:
            print(
                "Warning: ParamActor does not have 'action_parameters_passthrough_layer'. Skipping set_action_parameter_passthrough_weights.")

    def _seed(self, seed=None):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if seed is not None:  # np.random.RandomState wants None or int, not float
            self.np_random = np.random.RandomState(seed=int(seed))
        else:
            self.np_random = np.random.RandomState()

        if seed is not None:
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed(int(seed))

    def _ornstein_uhlenbeck_noise(self, all_action_parameters_numpy):  # Input is numpy array
        """ Continuous action exploration using an OrnsteinUhlenbeck process. """
        if not self.use_ornstein_noise or self.noise is None or self.action_parameter_size == 0:
            return all_action_parameters_numpy
        # Apply noise only to the actual parameters
        noise_sample = self.noise.sample()  # This is of size self.action_parameter_size
        # Ensure action_parameter_range_numpy matches the size of noise_sample
        return all_action_parameters_numpy + (noise_sample * self.action_parameter_range_numpy)

    def start_episode(self):
        if self.use_ornstein_noise and self.noise is not None:
            self.noise.reset()

    def end_episode(self):
        self._episode += 1
        '''
        ep = self._episode
        if self.epsilon_steps > 0:  # Avoid division by zero if epsilon_steps is 0
            if ep < self.epsilon_steps:
                self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                        ep / self.epsilon_steps)
            else:
                self.epsilon = self.epsilon_final
        else:
            self.epsilon = self.epsilon_final
        '''

    def act(self, state, _eval=False):  # Added _eval flag
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)  # Ensure float

            # Get all actual parameters from ParamActor
            # Output of actor_param is shape (self.action_parameter_size)
            all_actual_params_tensor = self.actor_param.forward(state_tensor.unsqueeze(0)).squeeze(
                0)  # Remove batch dim for processing

            # Epsilon-greedy for discrete action selection
            if not _eval and self.np_random.uniform() < self.epsilon:
                discrete_action = self.np_random.choice(self.num_actions)
                # For exploration, generate random parameters for actions that have them
                if self.action_parameter_size > 0:
                    all_actual_params_numpy = self.np_random.uniform(self.action_parameter_min_numpy,
                                                                     self.action_parameter_max_numpy)
                    all_actual_params_tensor = torch.from_numpy(all_actual_params_numpy).float().to(self.device)
                # else: all_actual_params_tensor remains (empty or as is if no params)
            else:
                # Select discrete action based on Q-values from QActor
                # QActor expects state and the concatenated actual parameters
                Q_a = self.actor.forward(state_tensor.unsqueeze(0),
                                         all_actual_params_tensor.unsqueeze(0))  # Add batch dim
                Q_a = Q_a.detach().cpu().numpy().flatten()  # Flatten to 1D array for argmax
                discrete_action = np.argmax(Q_a)

            # Noise addition (if not evaluating and noise is enabled)
            all_actual_params_numpy = all_actual_params_tensor.cpu().numpy()  # Convert to numpy for noise and slicing
            if not _eval and self.use_ornstein_noise and self.action_parameter_size > 0:
                all_actual_params_numpy = self._ornstein_uhlenbeck_noise(all_actual_params_numpy)

            # Clip parameters to their valid range
            if self.action_parameter_size > 0:
                all_actual_params_numpy = np.clip(all_actual_params_numpy,
                                                  self.action_parameter_min_numpy,
                                                  self.action_parameter_max_numpy)

            # Extract the specific parameters for the chosen discrete_action
            # self.action_parameter_offsets refers to positions in the concatenated vector of actual params
            # self.action_parameter_sizes refers to the size of params for each discrete action (can be 0)

            current_action_param_size = self.action_parameter_sizes[discrete_action]
            if current_action_param_size > 0:
                # Find the start index of this action's parameters in the concatenated actual_params vector
                # This requires mapping discrete_action index to its position if params were concatenated only for those >0
                # Simpler: use the offsets that are already designed for the concatenated vector.
                # The offsets are [0, P0_size, P0_size+P1_size, ... ]
                # Find which "block" discrete_action corresponds to in terms of non-zero parameters
                #param_idx_start = 0
                #params_seen_count = 0
                #actual_param_start_offset = -1

                # Iterate through actions to find the correct starting offset in the all_actual_params_numpy vector
                current_offset_in_concatenated_params = 0
                for i in range(discrete_action):
                    current_offset_in_concatenated_params += self.action_parameter_sizes[i]

                actual_param_end_offset = current_offset_in_concatenated_params + current_action_param_size

                params_for_chosen_action_numpy = all_actual_params_numpy[
                                                 current_offset_in_concatenated_params:actual_param_end_offset]

            else:  # Action has no parameters
                params_for_chosen_action_numpy = np.array([])

        # Returns: discrete_action_idx, params_for_THE_chosen_action (empty if none), all_actual_params_CONCATENATED_vector
        return discrete_action, params_for_chosen_action_numpy, all_actual_params_numpy

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        # grad is for the concatenated actual parameters (size self.action_parameter_size)
        # batch_action_indices are the discrete actions taken for each item in batch
        assert grad.shape[1] == self.action_parameter_size  # grad is (batch_size, self.action_parameter_size)

        grad = grad.cpu()
        if not inplace:
            grad = grad.clone()

        with torch.no_grad():
            # Build an 'action_map' for the elements of the concatenated parameter vector
            # E.g., if params_sizes = [1,0,1], concatenated_params has 2 elements.
            # 1st element belongs to action 0. 2nd element belongs to action 2.
            action_map = torch.zeros(self.action_parameter_size, dtype=torch.long)
            current_param_idx = 0
            for action_idx, param_size in enumerate(self.action_parameter_sizes):
                if param_size > 0:
                    action_map[current_param_idx: current_param_idx + param_size] = action_idx
                    current_param_idx += param_size

            # Tile this map for the batch
            # action_map_tiled has shape (batch_size, self.action_parameter_size)
            # Each row is action_map, e.g. [0, 0, 2] if action_parameter_sizes=[2,0,1]
            action_map_tiled = action_map.repeat(grad.shape[0], 1)  # grad.shape[0] is batch_size

            # batch_action_indices is (batch_size,). Unsqueeze to (batch_size,1) for broadcasting
            # We want to zero out grad[b, p] if param p does not belong to batch_action_indices[b]
            mask_to_zero = action_map_tiled != batch_action_indices.cpu().unsqueeze(1)
            grad[mask_to_zero] = 0.
        return grad.to(self.device)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # grad and vals are for the concatenated actual parameters
        # grad_type should be "action_parameters"
        if self.action_parameter_size == 0:  # No parameters to invert for
            return grad

        if grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        # Ensure tensors are on CPU for numpy-like operations if not already
        # Or ensure all ops are torch ops on the correct device
        # The PDQNAgent usually puts these on self.device already.
        # Let's assume grad, vals, max_p, min_p, rnge are already on self.device

        if not inplace:
            grad = grad.clone()

        # Index for positive gradients: grad > 0
        # For these, we want to scale by (max_p - vals) / rnge
        idx_pos_grad = grad > 0
        grad[idx_pos_grad] *= ((max_p - vals) / rnge)[idx_pos_grad]

        # Index for negative gradients: grad < 0 (actually <= 0 to cover all)
        # For these, we want to scale by (vals - min_p) / rnge
        idx_neg_grad = grad <= 0  # Using <= to cover zero grad case too, though it won't change
        grad[idx_neg_grad] *= ((vals - min_p) / rnge)[idx_neg_grad]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        # action is (discrete_action_idx, all_actual_params_vector)
        # next_action is (next_discrete_action_idx, next_all_actual_params_vector)
        discrete_act_idx, all_actual_params_vector = action

        # For replay buffer, store discrete_act_idx and all_actual_params_vector together
        action_bundle_for_memory = np.concatenate(([discrete_act_idx], all_actual_params_vector)).ravel()

        next_discrete_act_idx, next_all_actual_params_vector = next_action
        next_action_bundle_for_memory = np.concatenate(([next_discrete_act_idx], next_all_actual_params_vector)).ravel()

        self._step += 1
        self.replay_memory.append(state, action_bundle_for_memory, reward, next_state, terminal,
                                  next_action=next_action_bundle_for_memory)

        if len(self.replay_memory) >= self.batch_size and len(self.replay_memory) >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _optimize_td_loss(self):
        # Sample a batch from replay memory
        # Memory returns: states, actions_bundle, rewards, next_states, next_actions_bundle, terminals
        states, actions_bundle, rewards, next_states, _, terminals = self.replay_memory.sample(
            self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        # actions_bundle contains (discrete_action_idx, all_actual_params_vector)
        discrete_actions = torch.from_numpy(actions_bundle[:, 0]).long().to(self.device)  # Shape: (batch_size,)
        # all_actual_params are the concatenated parameters for actions that have them
        all_actual_params = torch.from_numpy(actions_bundle[:, 1:]).float().to(
            self.device)  # Shape: (batch_size, self.action_parameter_size)

        rewards = torch.from_numpy(rewards).float().to(self.device)  # Shape: (batch_size, 1)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        # next_actions_bundle contains (next_discrete_idx, next_all_actual_params)
        #next_discrete_actions = torch.from_numpy(next_actions_bundle[:, 0]).long().to(self.device)
        #next_all_actual_params = torch.from_numpy(next_actions_bundle[:, 1:]).float().to(self.device)

        terminals = torch.from_numpy(terminals).float().to(self.device)  # Shape: (batch_size, 1)

        # ---------------------- optimize Q-network (self.actor) ----------------------
        with torch.no_grad():
            # Predict Q'-values from target Q-network (self.actor_target)
            # For PDQN, the policy for next actions is deterministic from actor_param_target
            pred_next_actual_params = self.actor_param_target.forward(
                next_states)  # (batch_size, self.action_parameter_size)

            # QActor/MultiPassQActor needs state and concatenated actual params
            pred_Q_prime = self.actor_target.forward(next_states,
                                                     pred_next_actual_params)  # (batch_size, num_discrete_actions)

            # Select max Q' value among discrete actions for DDPG-style update, or use Q value of next_discrete_actions if SARSA-like
            # For PDQN, typically DDPG: max_a' Q'(s', a')
            Q_prime_max = torch.max(pred_Q_prime, 1, keepdim=True)[0]  # (batch_size, 1)

            # Compute the TD target
            y_expected = rewards + (1 - terminals) * self.gamma * Q_prime_max

        # Compute current Q-values using policy network (self.actor)
        # QActor/MultiPassQActor needs state and actual params corresponding to the *actions taken*
        q_values_all_actions = self.actor.forward(states, all_actual_params)  # (batch_size, num_discrete_actions)

        # Gather Q-values for the discrete actions that were actually taken
        # discrete_actions is (batch_size,), need (batch_size,1) for gather
        y_predicted = q_values_all_actions.gather(1, discrete_actions.unsqueeze(1))  # (batch_size, 1)

        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor-parameter network (self.actor_param) ----------------------
        self.actor_param_optimiser.zero_grad()

        # 1. Get current action parameters from the policy network (self.actor_param).
        # This tensor's graph is kept for the final backward pass to update actor_param's weights.
        current_policy_actual_params = self.actor_param.forward(
            states)  # Shape: (batch_size, self.action_parameter_size)

        # 2. Prepare these parameters to be treated as an input for dQ/dparams calculation.
        # We detach them from the graph connected to actor_param's weights for *this specific gradient calculation*,
        # but enable requires_grad to get gradients with respect to their values.
        params_for_dQ_dparams = current_policy_actual_params.detach().requires_grad_(True)

        # 3. Calculate Q-values using these "input-like" parameters and the main Q-network (self.actor).
        # This is the primary Q-value calculation needed for deriving delta_a.
        Q_val_for_dQ_dparams = self.actor.forward(states,
                                                  params_for_dQ_dparams)  # Shape: (batch_size, num_discrete_actions)

        # 4. Define the objective based on these Q-values, which will be used to get dQ/dparams.
        # `discrete_actions` is from the replay buffer.
        objective_q_values_for_grad = Q_val_for_dQ_dparams

        if self.weighted:
            with torch.no_grad():
                counts = Counter(discrete_actions.cpu().numpy())  # discrete_actions from replay buffer
                weights_np = np.array([counts.get(a, 0) / discrete_actions.shape[0] for a in range(self.num_actions)],
                                      dtype=np.float32)
                weights = torch.from_numpy(weights_np).float().to(self.device)
            objective_q_values_for_grad = weights.unsqueeze(0) * objective_q_values_for_grad
        elif self.average:
            objective_q_values_for_grad = objective_q_values_for_grad / self.num_actions

        # Determine the scalar objective whose gradient w.r.t. params_for_dQ_dparams is needed.
        if self.indexed:
            # Uses Q-values for discrete actions from the buffer, combined with parameters from current policy.
            scalar_objective_for_dQ_dparams = torch.mean(
                objective_q_values_for_grad.gather(1, discrete_actions.unsqueeze(1)))
        else:
            # Default: Sum of (potentially weighted/averaged) Q-values across all discrete actions.
            scalar_objective_for_dQ_dparams = torch.mean(torch.sum(objective_q_values_for_grad, dim=1))

        # 5. Calculate d(scalar_objective)/d(params_for_dQ_dparams)
        # This backward pass computes gradients ONLY w.r.t. params_for_dQ_dparams, not actor_param's weights directly yet.
        self.actor.zero_grad()
        scalar_objective_for_dQ_dparams.backward()

        # delta_a now holds d(ObjectiveQ)/d(params_values)
        delta_a = params_for_dQ_dparams.grad.data.clone()

        # 6. Apply custom gradient manipulations to delta_a
        if self.inverting_gradients and self.action_parameter_size > 0:
            # _invert_gradients needs (grad, values_of_params_grad_was_taken_wrt)
            delta_a = self._invert_gradients(delta_a, params_for_dQ_dparams.data,
                                             grad_type="action_parameters", inplace=True)

        if self.zero_index_gradients and self.action_parameter_size > 0: #这将确保你只更新连续参数的梯度，避免无关梯度的更新
            # If zero-indexing, it ideally needs discrete actions chosen by the *current* policy.
            # These Q-values are based on current policy parameters (params_for_dQ_dparams).
            with torch.no_grad():  # Getting actions for indexing shouldn't affect gradients
                current_policy_chosen_discrete_actions = torch.argmax(Q_val_for_dQ_dparams, dim=1)
            delta_a = self._zero_index_gradients(delta_a, current_policy_chosen_discrete_actions, inplace=True)
            # Note: If self.indexed was true above, `discrete_actions` from buffer were used.
            # For consistency, if zero_index_gradients is about aligning grad with specific actions,
            # using actions derived from Q_val_for_dQ_dparams (current policy's view) is more standard here.

        # 7. Backpropagate these modified gradients `delta_a` through the original `current_policy_actual_params`
        #    tensor, which IS connected to `self.actor_param`'s weights.
        # The negative sign is because optimizers minimize, and delta_a is dQ/dparams (we want to ascend Q).
        # 训练策略网络的目标是 最大化 Q 值，通过对动作参数的梯度手动计算得到优化方向；这时，需要用 output_tensor.backward(gradient=...) 给输出张量传递一个“外部梯度”，触发对参数的梯度计算(gradient=这里用来把动作梯度信号反传给产生动作的网络参数)；
        # delta_a 是由 Q 网络计算得到的 对这些动作参数的梯度，告诉我们动作参数应该如何调整才能提升 Q 值；
        current_policy_actual_params.backward(gradient=-delta_a)

        # 8. Clip gradients of self.actor_param's actual weights and perform optimizer step.
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        torch.save(self.actor.state_dict(), prefix + '_q_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_param_actor.pt')
        print(f'Models saved to {prefix}_q_actor.pt and {prefix}_param_actor.pt')

    def load_models(self, prefix):
        self.actor.load_state_dict(torch.load(prefix + '_q_actor.pt', map_location=self.device))
        self.actor_param.load_state_dict(torch.load(prefix + '_param_actor.pt', map_location=self.device))
        hard_update_target_network(self.actor, self.actor_target)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        print(f'Models loaded from {prefix}_q_actor.pt and {prefix}_param_actor.pt')

    # 【新增】保存完整的训练检查点
    def save_checkpoint(self, path, episode, total_steps,model_name='checkpoint.pt',mode="unsaved_data"):
        if not os.path.exists(path):
            os.makedirs(path)
        if mode == 'unsaved_data':
            checkpoint = {
                'step': self._step,
                'updates': self.updates,
                'total_steps': total_steps,
                'actor_state_dict': self.actor.state_dict(),
                'actor_param_state_dict': self.actor_param.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'actor_param_target_state_dict': self.actor_param_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimiser.state_dict(),
                'actor_param_optimizer_state_dict': self.actor_param_optimiser.state_dict(),
                'epsilon': self.epsilon
            }
        else:
            checkpoint = {
                'episode': episode,
                'step':self._step,
                'updates':self.updates,
                'total_steps': total_steps,
                'actor_state_dict': self.actor.state_dict(),
                'actor_param_state_dict': self.actor_param.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'actor_param_target_state_dict': self.actor_param_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimiser.state_dict(),
                'actor_param_optimizer_state_dict': self.actor_param_optimiser.state_dict(),
                'epsilon': self.epsilon,
                'replay_buffer_state': self.replay_memory.get_state()  # 保存经验回放池的状态
            }
        torch.save(checkpoint, os.path.join(path, model_name))
        print(f"在第 {episode} 回合，成功保存检查点到 '{path}'")

    # 【新增】加载训练检查点
    def load_checkpoint(self,path,model_file, load_type="train"):
        if load_type == "train":
            checkpoint_path = os.path.join(path, model_file)
            print('加载训练模型路径')
        else:
            checkpoint_path = os.path.join(path,model_file )#'checkpoint_simulatorV3.pt')
            print('加载测试模型:',model_file)
        if not os.path.exists(checkpoint_path):
            print(f"警告：在 '{path}' 未找到检查点文件。将从头开始。")
            raise ValueError("传入了不正确的路径")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 恢复所有网络权重
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_param.load_state_dict(checkpoint['actor_param_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.actor_param_target.load_state_dict(checkpoint['actor_param_target_state_dict'])

        # 恢复优化器状态
        self.actor_optimiser.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_param_optimiser.load_state_dict(checkpoint['actor_param_optimizer_state_dict'])

        # 恢复探索状态
        self.epsilon = checkpoint['epsilon']
        self._step = checkpoint['step']
        self.updates = checkpoint['updates']
        print('模型已成功加载')
        if load_type == "train":
            # 恢复经验回放池
            self.replay_memory.set_state(checkpoint['replay_buffer_state'])
            # 恢复训练进度
            episode = checkpoint.get('episode', 0)
            total_steps = checkpoint.get('total_steps', 0)
            print(f"成功从 '{path}' 加载检查点。将从第 {episode + 1} 回合继续。")
            print(f"恢复后，Epsilon 为 {self.epsilon:.4f}")
            print('训练模型加载成功')
            return episode, total_steps
        else:
            print('测试模型加载成功')
            return 0, 0