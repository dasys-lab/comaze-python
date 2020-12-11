from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional

import numpy as np 

import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions 

from comaze.agents.rl import AbstractOnPolicyRLAgent
from comaze.agents.utils import dict_encoded_pov_avail_moves_extract_exp_fn, discrete_direction_only_format_move_fn


class SimpleOnPolicyRLAgent(AbstractOnPolicyRLAgent):
  """
  Simple on-policy RL agents using PyTorch.
  
  Call init_rl_algo at the end of the init function.

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
    learning_rate: float=1e-4,
    discount_factor: float=0.99,
    num_actions: int=5,
    pov_shape: List[int]=[7,7,12],
    agent_order: int=0, 
    environment: Optional[gym.Env]=None,
    use_cuda: Optional[Any]=False,
    ) -> None:
    """
    Initializes the agent.
    """
    nn.Module.__init__(self=self)
    AbstractOnPolicyRLAgent.__init__(
      self=self,
      extract_exp_fn=dict_encoded_pov_avail_moves_extract_exp_fn, 
      format_move_fn=discrete_direction_only_format_move_fn,
      learning_rate=learning_rate,
      discount_factor=discount_factor,
      agent_order=agent_order,
      environment=environment,
    )

    self.num_actions = num_actions
    self.pov_shape = pov_shape
    self.use_cuda = use_cuda
    self.build_agent()
    
    self.init_rl_algo()
  
  @property
  def agent_id(self) -> str:
    return "simple_onpolicy_rlagent"
  
  def build_agent(self):
    self.embed_pov_size = 256
    self.embed_pov = nn.Sequential(
      nn.Conv2d(in_channels=self.pov_shape[-1], out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(512, self.embed_pov_size),
      nn.ReLU(),
    )
    
    self.embed_action_size = 128
    self.embed_action_space = nn.Linear(self.num_actions, self.embed_action_size)
    
    policy_input_size = self.embed_pov_size+self.embed_action_size
    self.policy = nn.Linear(policy_input_size, self.num_actions)
    
    if self.use_cuda:
      self.cuda()

  def get_formatted_inputs(self, obs):
    nobs = {}
    for k,v in obs.items():
      if 'pov' in k:
        # move channels around:
        assert len(v.shape)==3
        v = np.transpose(v, (2,0,1))
      nv = torch.from_numpy(v).unsqueeze(0).float()
      nobs[k] = nv.cuda() if self.use_cuda else nv
    return nobs

  def select_action(self, observation: Any) -> Dict[str, Any]:
    """
    Returns agent's action given `observation`.
    """

    obs = self.get_formatted_inputs(observation)

    pov_input = obs["encoded_pov"]
    action_space = obs["available_moves"]
    
    pov_emb = self.embed_pov(pov_input)
    action_emb = self.embed_action_space(action_space)
    
    pov_action_emb = torch.cat((pov_emb, action_emb), dim=1)
    action_pred = self.policy(pov_action_emb)
    
    action_prob = F.softmax(action_pred, dim = -1)  
    avail_action_prob = action_prob * obs["available_moves"]
    dist = distributions.Categorical(avail_action_prob)
    action = dist.sample()
    log_prob_action = dist.log_prob(action)

    action_dict = {
      "action": action.item(),
      "log_prob_action": log_prob_action
    }

    return action_dict