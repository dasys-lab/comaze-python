import random
from typing import Callable
import pandas as pd 

from functools import partial
from tqdm import tqdm 

from comaze.env import TwoPlayersCoMazeGym
from comaze.agents import AbstractAgent, SimpleOnPolicyRLAgent


def two_players_environment_loop(
    agent1: AbstractAgent,
    agent2: AbstractAgent,
    environment,
    max_episode_length,
):
  """
  Loop runner for the environment.
  """

  # Setup environment.
  #environment = TwoPlayersCoMazeGym(**environment_kwargs)
  state = environment.reset()

  # Initialize agents.
  agent1.set_environment(environment=environment, agent_order=0)
  agent2.set_environment(environment=environment, agent_order=1)

  # Book-keeping.
  t = 0
  done = False
  trajectory = list()

  ebar = tqdm(total=max_episode_length, position=1)
  while not done and t<=max_episode_length:
    ebar.update(1)
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
    """
    if t%2 == 0:
      agent1.update(move, next_state, reward, done)
    else:
      agent2.update(move, next_state, reward, done)
    """
    if t==max_episode_length:
      done = True
      reward = -1
    
    for agent in [agent1, agent2]:
      agent.update(move, next_state, reward, done)

    # Book-keeping.
    t = t + 1
    state = next_state
  

  # Dump logs.
  pd.DataFrame(trajectory).to_csv("{}-{}.csv".format(
      agent1.agent_id, agent2.agent_id)
  )


def test_training_simple_on_policy_rl_agent():
    agent1 = SimpleOnPolicyRLAgent( 
      learning_rate=1e-4,
      discount_factor=0.99,
      num_actions=5,
      pov_shape=[7,7,12],
    )

    agent2 = SimpleOnPolicyRLAgent( 
      learning_rate=1e-4,
      discount_factor=0.99,
      num_actions=5,
      pov_shape=[7,7,12],
    )

    max_episode_length = 50
    nbr_training_episodes = 1000
    verbose = False 

    tbar = tqdm(total=nbr_training_episodes, position=0)
    for episode in range(nbr_training_episodes):
      tbar.update(1)
      environment_kwargs = {
          "level":"1",
          "verbose":verbose,
      }
      environment = TwoPlayersCoMazeGym(**environment_kwargs)

      two_players_environment_loop(
          agent1=agent1,
          agent2=agent2,
          environment=environment,
          max_episode_length=max_episode_length,
      )

if __name__ == "__main__":
    test_training_simple_on_policy_rl_agent()
