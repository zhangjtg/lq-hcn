# agents/memory/memory.py
import numpy as np
import random


class Memory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, observation_shape, action_shape, next_actions=False, seed=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            observation_shape (tuple): shape of an observation
            action_shape (tuple): shape of an action
            seed (int): random seed
        """
        self.memory = np.empty(buffer_size, dtype=[
            ("observation", np.float32, observation_shape),
            ("action", np.float32, action_shape),  # For PDQN, action includes discrete + continuous params
            ("reward", np.float32),
            ("next_observation", np.float32, observation_shape),
            ("terminal", bool)
        ])
        if next_actions:  # For SARSA style updates, not strictly needed if PDQN recalculates next_action
            self.memory = np.empty(buffer_size, dtype=[
                ("observation", np.float32, observation_shape),
                ("action", np.float32, action_shape),
                ("reward", np.float32),
                ("next_observation", np.float32, observation_shape),
                ("next_action", np.float32, action_shape),
                ("terminal", bool)
            ])

        self.buffer_size = buffer_size
        self.action_shape = action_shape
        self._idx = 0
        self._size = 0
        self.next_actions_stored = next_actions

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.random_machine = np.random.RandomState(seed=seed)

    def append(self, observation, action, reward, next_observation, terminal, next_action=None):
        """Add a new experience to memory."""
        idx = self._idx

        if self.next_actions_stored:
            if next_action is None:
                # Provide a default zero action if next_action is not given but expected
                next_action = np.zeros(self.action_shape, dtype=np.float32)
            self.memory[idx] = (observation, action, reward, next_observation, next_action, terminal)
        else:
            self.memory[idx] = (observation, action, reward, next_observation, terminal)

        self._idx = (idx + 1) % self.buffer_size
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size, random_machine=None):
        """Randomly sample a batch of experiences from memory."""
        if random_machine is None:
            random_machine = self.random_machine

        # Ensure we only sample from filled part of the buffer
        indices = random_machine.choice(self._size, size=batch_size, replace=False)

        observations = self.memory["observation"][indices]
        actions = self.memory["action"][indices]
        rewards = self.memory["reward"][indices].reshape(-1, 1)
        next_observations = self.memory["next_observation"][indices]
        terminals = self.memory["terminal"][indices].reshape(-1, 1)

        if self.next_actions_stored:
            next_actions_sample = self.memory["next_action"][indices]
            return observations, actions, rewards, next_observations, next_actions_sample, terminals
        else:
            return observations, actions, rewards, next_observations, terminals

    def __len__(self):
        """Return the current size of internal memory."""
        return self._size

    def get_state(self):
        """【新增】获取经验池的完整状态用于保存。"""
        return {
            'memory': self.memory,
            '_idx': self._idx,
            '_size': self._size
        }

    def set_state(self, state):
        """【新增】从一个状态字典中恢复经验池。"""
        self.memory = state['memory']
        self._idx = state['_idx']
        self._size = state['_size']
        print(f"经验池已恢复，当前包含 {self._size} 条经验。")