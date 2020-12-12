import random
from typing import Any
from typing import Dict
from .abstract_agent import AbstractAgent


class RuleBasedAgent(AbstractAgent):
  @property
  def agent_id(self) -> str:
    """
    Please provide the teamID you have been given at registration time,
    and add a prefix of your choice.
    The resulting form should be:
    return "prefix-teamID"
    """
    return "rulebased-testTeamID"
  
  def select_action(self, observation: Any) -> Dict[str, Any]:
    """
    Template rule-based agent: Nearest Goal Agent
    Choose a nearest goal, see if one of your actions can get you there, if so take that action

    """
    action_space = observation["currentPlayer"]["directions"]

    goals_pos = [goal['position'] for goal in observation['unreachedGoals']]
    agent_pos = observation['agentPosition']
    
    goal_diffs = [
      (goal['x'] - agent_pos['x'], goal['y'] - agent_pos['y'])
      for goal in goals_pos
    ]
    goal_dists = [
      abs(diff[0])+abs(diff[1]) 
      for diff in goal_diffs
    ]
    nearest_goal = goal_dists.index(min(goal_dists)) 

    print(f'Nearest goal is {observation["unreachedGoals"][nearest_goal]}')
    print(f'Nearest goal diff {goal_diffs[nearest_goal]}')

    move_x, move_y = goal_diffs[nearest_goal]

    if 'LEFT' in action_space and move_x < 0:
      direction = 'LEFT'
    elif 'RIGHT' in action_space and move_x > 0:
      direction = 'RIGHT'
    elif 'UP' in action_space and move_y < 0:
      direction = 'UP'
    elif 'DOWN' in action_space and move_y > 0:
      direction = 'DOWN'
    else:
      direction = 'SKIP'
      
    action_dict = {
        "action": {"direction":direction},
    }
    return action_dict
    