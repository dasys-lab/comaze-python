import random
from typing import Callable
import pandas as pd 

from functools import partial
from tqdm import tqdm 

from comaze.env import TwoPlayersCoMazeGym
from comaze.agents import AbstractAgent, RandomAgent, RuleBasedAgent


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
  state = environment.reset()

  # Book-keeping.
  t = 0
  done = False
  trajectory = list()
  cum_reward = 0

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

    if t==max_episode_length:
      done = True
      reward += -1

    for agent in [agent1, agent2]:
      agent.update(move, next_state, reward, done)

    # Book-keeping.
    trajectory.append((t, state, move, reward, next_state, done, info))

    cum_reward += reward
    t = t + 1
    state = next_state
  

  # Dump logs.
  pd.DataFrame(trajectory).to_csv("{}-{}.csv".format(
      agent1.agent_id, agent2.agent_id)
  )

  return cum_reward, trajectory

def test_rule_based_agent():
  use_cuda = False 
  sparse_reward = False

  agent1 = RandomAgent()

  agent2 = RuleBasedAgent()

  logging_path = './test_training.log'
  logger = SummaryWriter(logging_path)

  max_episode_length = 50
  nbr_training_episodes = 1000
  verbose = False

  environment_kwargs = {
    "level":"1",
    "sparse_reward":sparse_reward,
    "verbose":verbose,
    "agent_names": [
      agent1.agent_id,
      agent2.agent_id,
    ]
  }
  
  environment = TwoPlayersCoMazeGym(**environment_kwargs)

  episode_cum_reward, trajectory = two_players_environment_loop(
    agent1=agent1,
    agent2=agent2,
    environment=environment,
    max_episode_length=max_episode_length,
  )

  agent1.save()
  agent2.save()

if __name__ == "__main__":
  test_rule_based_agent()
