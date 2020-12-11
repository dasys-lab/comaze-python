import random
from typing import Callable
import pandas as pd 

from functools import partial

from comaze.env import TwoPlayersCoMazeGym
from comaze.agents import AbstractAgent, SimpleOnPolicyRLAgent
from comaze.agents.utils import dict_encoded_pov_avail_moves_extract_exp_fn, discrete_direction_only_format_move_fn

def two_players_environment_loop(
    agent1_fn: Callable[..., AbstractAgent],
    agent2_fn: Callable[..., AbstractAgent],
    environment, #**environment_kwargs,
):
  """
  Loop runner for the environment.
  """

  # Setup environment.
  #environment = TwoPlayersCoMazeGym(**environment_kwargs)
  state = environment.reset()

  # Initialize agents.
  agent1 = agent1_fn(environment=environment, agent_order=0)
  agent2 = agent2_fn(environment=environment, agent_order=1)

  # Book-keeping.
  t = 0
  done = False
  trajectory = list()

  while not done:
    # Turn-based game.
    if t%2 == 0:
      move = agent1.select_move(state)
    else:
      move = agent2.select_move(state)
  
    # Progress simulation.
    next_state, reward, done, info = environment.step(move)

    # Used for logging.
    trajectory.append((t, state, move, reward, next_state, done, info))

    # Agent internals.
    if t%2 == 0:
      agent1.update(move, next_state, reward, done)
    else:
      agent2.update(move, next_state, reward, done)
    
    # Book-keeping.
    t = t + 1
    state = next_state
  
  # Dump logs.
  pd.DataFrame(trajectory).to_csv("{}-{}.csv".format(
      agent1.agent_id, agent2.agent_id)
  )


def test_dict_encoded_pov_avail_move_exp_utils():
    agent1_fn = partial(
      SimpleOnPolicyRLAgent, 
      learning_rate=1e-4,
      discount_factor=0.99,
      num_actions=5,
      pov_shape=[7,7,12],
      #extract_exp_fn=dict_encoded_pov_avail_moves_extract_exp_fn,
      #format_move_fn=discrete_direction_only_format_move_fn,
    )
    agent2_fn = partial(
      SimpleOnPolicyRLAgent, 
      learning_rate=1e-4,
      discount_factor=0.99,
      num_actions=5,
      pov_shape=[7,7,12],
      #extract_exp_fn=dict_encoded_pov_avail_moves_extract_exp_fn,
      #format_move_fn=discrete_direction_only_format_move_fn,
    )

    environment_kwargs = {
        "level":"1",
    }
    environment = TwoPlayersCoMazeGym(**environment_kwargs)

    two_players_environment_loop(
        agent1_fn=agent1_fn,
        agent2_fn=agent2_fn,
        environment=environment
    )

if __name__ == "__main__":
    test_dict_encoded_pov_avail_move_exp_utils()
