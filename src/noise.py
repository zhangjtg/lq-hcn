# agents/utils/noise.py
import numpy as np
import copy

class OrnsteinUhlenbeckActionNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, random_machine, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.random_machine = random_machine
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.random_machine.randn(self.size)
        self.state = x + dx
        return self.state