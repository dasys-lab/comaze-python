import random
from .abstract_agent import AbstractAgent, Observation, Action


class RandomAgent(AbstractAgent):
  @property
  def agent_id(self) -> str:
    return "random"
  
  def select_action(self, observation: Observation) -> Action:
    """Randomly selects an action out of the legal ones."""
    return random.choice(self._environment.action_space[self._agent_order])