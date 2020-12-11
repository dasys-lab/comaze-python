import random
from typing import Any
from typing import Dict
from .abstract_agent import AbstractAgent, Observation, Action


class RandomCommunicatingAgent(AbstractAgent):
  @property
  def agent_id(self) -> str:
    return "random"
  
  def select_action(self, observation: Any) -> Dict[str, Any]:
    """
    Randomly selects an action out of all the possible ones.
    """
    direction = random.choice(self.actionId2action+["SKIP"])
    action_dict = {
        "action": {"direction":direction},
    }
    
    if direction == "SKIP":
      return action_dict

    message_id = random.choice(list(self.id2token.keys()))
    if message_id == 0:
      return action_dict

    message = self.id2token[message_id]
    action_dict["action"]["symbol_Message"] = message 

    return action_dict