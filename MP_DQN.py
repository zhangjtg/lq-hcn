# agents/pdqn_multipass_custom.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PDQN import PDQNAgent, ParamActor  # Use the custom PDQNAgent
from network_utils import hard_update_target_network  # Adjusted import


class MultiPassQActor(nn.Module):
    def __init__(self, state_size, action_size, action_parameter_size_list, hidden_layers=(100,),
                 output_layer_init_std=None, activation="relu", device='cpu', **kwargs):
        super().__init__()
        self.device = torch.device(device)  # Ensure device is a torch.device object
        self.state_size = state_size
        self.action_size = action_size  # Num discrete actions
        self.action_parameter_size_list = np.array(action_parameter_size_list, dtype=int)

        # Total dimension of the concatenated actual parameters (output of ParamActor)
        self.param_actor_output_size = int(np.sum(self.action_parameter_size_list))

        self.activation = activation

        # Create layers
        self.layers = nn.ModuleList()
        # Input to the NN layers is state + block for concatenated actual parameters
        inputSize = self.state_size + self.param_actor_output_size

        lastHiddenLayerSize = inputSize
        if hidden_layers is not None and len(hidden_layers) > 0:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        # Else, if no hidden layers, lastHiddenLayerSize remains inputSize.

        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))  # Outputs Q-value for each discrete action

        # Initialise layer weights
        for i in range(0, len(self.layers) - 1):  # For hidden layers
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)

        # For the output layer
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        else:  # A common default if not specified
            nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        # Offsets within the concatenated *actual* parameter vector (output of ParamActor)
        self.param_offsets = self.action_parameter_size_list.cumsum()
        self.param_offsets = np.insert(self.param_offsets, 0, 0)
        # E.g., if size_list = [1,1,0,0], param_offsets = [0,1,2,2,2]

    def forward(self, state, action_parameters):
        # state: (batch_size, state_size)
        # action_parameters: (batch_size, self.param_actor_output_size), from ParamActor

        negative_slope = 0.01
        batch_size = state.shape[0]

        # Ensure input tensors are on the correct device
        state = state.to(self.device)
        if self.param_actor_output_size > 0:
            action_parameters = action_parameters.to(self.device)

        # 1. Prepare the input tensor for the neural network.
        # It will be of shape (action_size * batch_size, state_size + param_actor_output_size)

        if self.param_actor_output_size > 0:
            # Case 1: There are continuous parameters for at least one action.
            # Create a base by concatenating state with a block of zeros for parameters.
            zeros_for_params = torch.zeros_like(action_parameters, device=self.device)
            processed_x = torch.cat((state, zeros_for_params), dim=1)

            # Repeat this base for each discrete action "pass".
            # Order of repeat: [S_batch_for_A0, S_batch_for_A1, ...]
            processed_x = processed_x.repeat(self.action_size, 1)

            # Fill in the actual parameters from `action_parameters` into the correct slots.
            for a_idx in range(self.action_size):  # For each discrete action "pass"
                if self.action_parameter_size_list[a_idx] > 0:  # If this action 'a_idx' has parameters
                    # Slice of `action_parameters` relevant to this action 'a_idx'
                    param_start_offset_in_ap = self.param_offsets[a_idx]
                    param_end_offset_in_ap = self.param_offsets[a_idx + 1]
                    params_to_insert = action_parameters[:, param_start_offset_in_ap:param_end_offset_in_ap]

                    # Rows in `processed_x` for this action pass
                    rows_start = a_idx * batch_size
                    rows_end = (a_idx + 1) * batch_size

                    # Columns in `processed_x` (within the parameter block) where these params go
                    col_start_in_param_block = self.state_size + param_start_offset_in_ap
                    col_end_in_param_block = self.state_size + param_end_offset_in_ap

                    processed_x[rows_start:rows_end,
                    col_start_in_param_block:col_end_in_param_block] = params_to_insert
        else:
            # Case 2: No parameters at all in any action (self.param_actor_output_size == 0).
            # The input to the network layers is just the repeated states.
            processed_x = state.repeat(self.action_size, 1)  # Shape: (action_size * batch_size, state_size)

        # 2. Pass the prepared batch through the network layers
        num_hidden_layers = len(self.layers) - 1  # Number of hidden layers
        for i in range(num_hidden_layers):
            layer = self.layers[i]
            if self.activation == "relu":
                processed_x = F.relu(layer(processed_x))
            elif self.activation == "leaky_relu":
                processed_x = F.leaky_relu(layer(processed_x), negative_slope)
            else:
                raise ValueError(f"Unknown activation function {self.activation}")

        # Output layer
        Q_all_passes_all_actions = self.layers[-1](processed_x)
        # Shape: (action_size * batch_size, self.action_size)

        # 3. Extract the Q-value for the specific action corresponding to each pass
        # For the block of rows from `rows_start` to `rows_end` (which represents action pass `a_idx`),
        # we need the Q-value for action `a_idx` itself (i.e., column `a_idx`).
        Q_output_list = []
        for a_idx in range(self.action_size):
            rows_start = a_idx * batch_size
            rows_end = (a_idx + 1) * batch_size

            # Q_all_passes_all_actions[rows_start:rows_end, a_idx] gives Q(state_batch, action=a_idx, params_for_a_idx)
            q_for_action_a_pass = Q_all_passes_all_actions[rows_start:rows_end, a_idx]  # Shape: (batch_size,)
            Q_output_list.append(q_for_action_a_pass.unsqueeze(1))  # Shape: (batch_size, 1)

        Q_final_for_batch = torch.cat(Q_output_list, dim=1)  # Shape: (batch_size, self.action_size)
        return Q_final_for_batch


class MultiPassPDQNAgent(PDQNAgent):  # Inherits from our modified PDQNAgent
    NAME = "Multi-Pass P-DQN Agent"

    def __init__(self,
                 *args,
                 **kwargs):  # observation_space, action_space, etc. are in args or kwargs

        # Extract actor_kwargs for MultiPassQActor, ensure 'device' is passed if MultiPassQActor expects it
        actor_kwargs_from_input = kwargs.get('actor_kwargs', {})

        # PDQNAgent constructor will handle observation_space, action_space, device etc.
        super().__init__(*args, **kwargs)  # Calls PDQNAgent.__init__

        # Override self.actor and self.actor_target with MultiPassQActor
        # self.action_parameter_sizes is already correctly set by PDQNAgent.__init__ for [1,1,0,0] like params
        # MultiPassQActor needs this list of sizes.
        self.actor = MultiPassQActor(self.observation_space.shape[0],
                                     self.num_actions,
                                     self.action_parameter_sizes,  # This is the list [1,1,0,0]
                                     device=self.device,  # Pass device
                                     **actor_kwargs_from_input).to(self.device)
        self.actor_target = MultiPassQActor(self.observation_space.shape[0],
                                            self.num_actions,
                                            self.action_parameter_sizes,
                                            device=self.device,  # Pass device
                                            **actor_kwargs_from_input).to(self.device)

        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        # Re-initialize optimiser for the new actor
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)