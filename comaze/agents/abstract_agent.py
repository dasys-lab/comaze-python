import abc
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional

import gym

from comaze.agents.utils import dummy_extract_exp_fn, dummy_format_move_fn

# Type definitions.
Observation = Dict[str, Any]
Action = Dict[str, str]

class AbstractAgent(abc.ABC):
  """
  Base class for agents.
  
  You need to subclass this and make sure to implement:
    - agent_id: Agent's unique id.
    - select_action: Agent's action selection logic.
  """

  def __init__(
    self, 
    extract_exp_fn: Callable[..., Any]=dummy_extract_exp_fn, 
    format_move_fn: Callable[..., Dict[str,str]]=dummy_format_move_fn,
    agent_order: int=0, 
    environment: Optional[gym.Env]=None, 
    ) -> None:
    """
    Initializes the agent.
    """
    self.actionId2action =  ["LEFT", "RIGHT", "UP", "DOWN"]
    self.id2token = {
      0:"empty", 
      1:"Q", 
      2:"W", 
      3:"E", 
      4:"R", 
      5:"T", 
      6:"Y", 
      7:"U", 
      8:"I", 
      9:"O", 
      10:"P"
    }
    
    self._environment = environment
    assert agent_order in (0, 1)
    self._agent_order = agent_order

    self.extract_exp_fn = extract_exp_fn
    self.format_move_fn = format_move_fn

    self.bookkeeping_dict = {}

  def set_environment(self, environment: gym.Env, agent_order: int):
    self._environment = environment
    assert agent_order in (0, 1)
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

  def select_move(self, observation: Observation) -> Action:
    """
    Returns agent's move in server-friendly format, 
    given a server-friendly-formatted `observation`.
    Use the extract_exp_fn and format_move_fn to accomodate the agent.
    """
    self.bookkeeping_dict["server-observation"] = observation
    obs = self.extract_exp_fn(observation, None)
    self.bookkeeping_dict["agent-observation"] = obs
    action_dict = self.select_action(obs)
    self.bookkeeping_dict["action_dict"] = action_dict
    formatted_move = self.format_move_fn(
      action_dict.get("action", None)
    )
    self.bookkeeping_dict["move"] = formatted_move
    return formatted_move

  @abc.abstractmethod
  def select_action(self, observation: Any) -> Dict[str, Any]:
    """
    Returns agent's action given `observation`.
    """