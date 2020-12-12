import abc
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional

import pickle
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

    self.extract_exp_fn = extract_exp_fn
    self.format_move_fn = format_move_fn
    
    self.bookkeeping_dict = {}

  @property
  @abc.abstractproperty
  def agent_id(self) -> str:
    """
    Please provide the teamID you have been given at registration time,
    and add a prefix of your choice.
    The resulting form should be:
    return "prefix-teamID"
    """

  def save(self):
    try:
      save_path = f"./{self.agent_id}.player"
      pickle.dump(self, open(save_path, 'wb'))
      print(f"Agent saved successfully at {save_path} .")
      return True
    except Exception as e:
      print(e)
      print("Agent was NOT save, please investigate:")
      import ipdb; ipdb.set_trace()

    return False

  def update(self, last_action, new_observation, reward, done) -> None:
    """
    Optional callback update function after env.step().
    """
    pass
  
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