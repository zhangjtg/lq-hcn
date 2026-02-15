# agents/agent.py
class Agent(object):
    """Base class for all Agent classes."""

    NAME = "Base Agent"

    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

    def __str__(self):
        return "<" + self.NAME + ">"

    def act(self, observation):
        """Return action to be taken in response to observation."""
        raise NotImplementedError()

    def start_episode(self):
        """Reset agent for a new episode."""
        pass

    def end_episode(self):
        """Signal the end of an episode."""
        pass

    def step(self, observation, action, reward, next_observation, next_action, terminal):
        """Provide the agent with the consequences of its action."""
        raise NotImplementedError()

    def save_models(self, prefix):
        """Save models to files."""
        raise NotImplementedError()

    def load_models(self, prefix):
        """Load models from files."""
        raise NotImplementedError()