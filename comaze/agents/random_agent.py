import random
from typing import Any
from typing import Dict
from .abstract_agent import AbstractAgent, Observation, Action


class RandomAgent(AbstractAgent):
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
    return action_dict
    #return {"direction":random.choice(self._environment.action_space[self._agent_order])}