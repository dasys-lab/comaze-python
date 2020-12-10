import abc
from typing import Any
from typing import Dict

import gym

# Type definitions.
Observation = Dict[str, Any]
Action = str

class AbstractAgent(abc.ABC):
  """
  Base class for agents.
  
  You need to subclass this and make sure to implement:
    - agent_id: Agent's unique id.
    - select_action: Agent's action selection logic.
  """

  def __init__(self, environment: gym.Env, agent_order: int) -> None:
    """
    Initializes the agent.
    """
    assert agent_order in (0, 1)

    self._environment = environment
    self._agent_order = agent_order

  def update(self, last_action, new_observation, reward, done) -> None:
    """
    Optional callback update function after env.step().
    """
    pass

  @property
  @abc.abstractproperty
  def agent_id(self) -> str:
    """
    User-defined agent unique idea (try to be creative to avoid collisions).
    """

  @abc.abstractmethod
  def select_action(self, observation: Observation) -> Action:
    """
    Returns agent's action given `observation`.
    """