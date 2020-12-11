from typing import Any
from typing import Dict
from typing import Callable

import gym
import torch 
import torch.nn as nn

from comaze.agents import AbstractAgent
from comaze.agents.utils import dict_encoded_pov_avail_moves_extract_exp_fn, discrete_direction_only_format_move_fn

# Type definitions.
Observation = Dict[str, Any]
Action = Dict[str, str]



class AbstractOnPolicyRLAgent(AbstractAgent, nn.Module):
  """
  Base class for on-policy RL agents using PyTorch.
  
  You need to subclass this and make sure to call init_rl_algo
  at the end of the init function.

  The output of select_action must be a dictionnary containing:
    - "action": the actual action that needs to be transformed 
                using the format_move_fn function.
    - "log_prob_action": the log likelihood over the action
                          distribution. 
  
  Note the default extract_exp_fn and format_move_fn functions.
  They are the minimum to allow any learning to take place.

  As AbstractAgent requests it, you also need to implement:
    - agent_id: Agent's unique id.
    - select_action: Agent's action selection logic.
  """

  def __init__(
    self, 
    environment: gym.Env, 
    agent_order: int, 
    extract_exp_fn: Callable[..., Any]=dict_encoded_pov_avail_moves_extract_exp_fn, 
    format_move_fn: Callable[..., Dict[str,str]]=discrete_direction_only_format_move_fn,
    learning_rate: float=1e-4,
    discount_factor: float=0.99
    ) -> None:
    """
    Initializes the agent.
    """
    super(AbstractOnPolicyRLAgent, self).__init__(
      environment=environment,
      agent_order=agent_order,
      extract_exp_fn=extract_exp_fn,
      format_move_fn=format_move_fn
    )

    self.learning_rate = learning_rate
    self.discount_factor = discount_factor

    self._episode_reset()

  def _episode_reset(self):
    """
    Callback made at the end of each episode 
    in order to be able to accumulate relevant 
    values in the next episode. 
    """
    self.episode_log_prob_actions = []
    self.episode_rewards = []
  
  def _calculate_returns(self, rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns  

  def init_rl_algo(self):
    self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

  def optimize(self):

    log_prob_actions = torch.cat(self.episode_log_prob_actions)
    returns = self._calculate_returns(self.episode_rewards, self.discount_factor).detach()
    loss = - (returns * log_prob_actions).sum()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    print(f'Loss {loss} :: EP reward {sum(self.episode_rewards)}')

    # reset:
    self._episode_reset()
    
  def update(self, last_action, new_observation, reward, done) -> None:
    """
    Updates the agent in an on-policy fashion.
    Callback made after env.step().
    """
    log_prob_action = self.bookkeeping_dict["action_dict"].get("log_prob_action", None)
    if log_prob_action is None:
      raise Exception("You must provide an entry 'log_prob_action' in the output dict of select_action.")
    self.episode_log_prob_actions.append(log_prob_action)
    self.episode_rewards.append(reward)

    if done:
      self.optimize()
    