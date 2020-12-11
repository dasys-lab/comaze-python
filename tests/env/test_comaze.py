import random
from typing import Callable
import pandas as pd 

from comaze.env import TwoPlayersCoMazeGym
from comaze.agents import AbstractAgent, RandomAgent, RandomCommunicatingAgent


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
  agent1 = agent1_fn(environment, agent_order=0)
  agent2 = agent2_fn(environment, agent_order=1)

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


def test_TwoPlayersCoMazeGym():
    agent1_fn = RandomAgent
    agent2_fn = RandomCommunicatingAgent
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
    test_TwoPlayersCoMazeGym()
