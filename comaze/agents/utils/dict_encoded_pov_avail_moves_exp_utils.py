import time
import numpy as np
import gym
from gym.spaces import Discrete


class DictEncodedPOVAvailMovesExpExtractorClass(object):
  def __init__(self, options={}):
    # We cannot initialise it as a normal wrapper since we have no environment to wrap...
    self.nb_directions = 4
    self.actionId2action =  ["LEFT", "RIGHT", "UP", "DOWN"]
    self.move2moveId =  {"LEFT":0, "RIGHT":1, "UP":2, "DOWN":3}
  
    # Observation Space:
    ## encoded_pov_space: the depth channel is a one-hot encoding of the tile nature:
    self.tile2id = {}
    self.nb_different_tile = 3       #background, time bonus, and agent
    self.tile2id["background"]= 0
    self.tile2id["agent"]= 1
    self.tile2id["time_bonus"]= 2
    self.nb_different_tile += 4+1    #4 unreached goals + 1 reached goal.
    self.goalEnum2id = {"RED":3, "BLUE":4, "GREEN":5, "YELLOW":6, 'reached_goal':7}
    self.tile2id["goal_1"]= 3
    self.tile2id["goal_2"]= 4
    self.tile2id["goal_3"]= 5
    self.tile2id["goal_4"]= 6
    self.tile2id["reached_goal"]= 7
    self.nb_different_tile += 4      # wall in any of the 4 directions.
    self.wallDirectionEnum2id = {"LEFT":8, "RIGHT":9, "UP":10, "DOWN":11}
    self.tile2id["wall_left"]= 8
    self.tile2id["wall_right"]= 9
    self.tile2id["wall_up"]= 10
    self.tile2id["wall_down"]= 11
    
    ## available_action_space:
    self.nb_possible_moves = self.nb_directions+1
    # SKIP...
    ##

  def _encode_game(self, game):
    grid = np.zeros(
      (game["config"]["arenaSize"]["x"], game["config"]["arenaSize"]["y"], self.nb_different_tile),
      dtype=np.int64,
    )
    for x in range(grid.shape[0]):
      for y in range(grid.shape[1]):
        grid[x][y][0] = 1
    
    # Agent:
    agent_x = game["agentPosition"]["x"]
    agent_y = game["agentPosition"]["y"]
    grid[agent_x, agent_y, 0] = 0
    grid[agent_x, agent_y, self.tile2id["agent"]] = 1
    
    # Goals:
    goals = game["config"]["goals"]
    unreached_goals = game["unreachedGoals"]
    for goal in goals:
      gx, gy = goal["position"]["x"], goal["position"]["y"]
      goal_id = self.goalEnum2id[goal["color"]]
      if goal not in unreached_goals:
        goal_id = self.goalEnum2id["reached_goal"]
      grid[gx, gy, 0] = 0
      grid[gx, gy, goal_id] = 1
    
    # Walls?
    walls = game["config"]["walls"]
    for wall in walls:
      wx, wy = wall["position"]["x"], goal["position"]["y"]
      wall_id = self.wallDirectionEnum2id[wall["direction"]]
      grid[wx, wy, 0] = 0
      grid[wx, wy, wall_id] = 1
    
    return grid

  def _get_available_moves(self, game):
    current_player_available_moves = game["currentPlayer"]["actions"]
    a_moves = np.zeros(self.nb_possible_moves)
    # SKIP alwasy available:
    a_moves[-1] = 1
    for move in current_player_available_moves:
      if move == "SKIP":    continue
      move_id = self.move2moveId[move]
      a_moves[move_id] = 1
    return a_moves

  def extract_exp(self, game, player):
    obs = {}
    obs["encoded_pov"] = self._encode_game(game=game)
    
    obs["available_moves"] = self._get_available_moves(game=game)
    
    return obs 

    """
    if game["state"]["won"]:
      reward = 1
    elif game["state"]["lost"]:
      reward = -1
    else:
      reward = 0

    done = game["state"]["over"]

    infos = None

    return obs, reward, done, infos
    """

DictEncodedPOVAvailMovesExpExtractor = DictEncodedPOVAvailMovesExpExtractorClass()
def dict_encoded_pov_avail_moves_extract_exp_fn(game, player):
  """
  Provided a game state and player information, 
  this function extracts an experience tuple, i.e.
  (observation, reward, done, infos) in a typical RL fashion.
  """
  global DictEncodedPOVAvailMovesExpExtractor
  return DictEncodedPOVAvailMovesExpExtractor.extract_exp(game, player)


class DiscreteDirectionOnlyFormatorClass(object):
  def __init__(self, options={}):
    self.nb_directions = 4
    self.moveId2move =  ["LEFT", "RIGHT", "UP", "DOWN", "SKIP"]
    
    # Action Space:
    self.nb_possible_actions = self.nb_directions+1
    # +1 accounts for the SKIP action...
    self.action_space = Discrete(self.nb_possible_actions)

  def format_move(self, action):
    if not self.action_space.contains(action):
      raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

    original_action_direction = self.moveId2move[action]

    rd = {
      'direction':original_action_direction,
      'symbol_Message':None
    }

    return rd 

DiscreteDirectionOnlyFormator = DiscreteDirectionOnlyFormatorClass()
def discrete_direction_only_format_move_fn(action):
  """
  Provided an RL-based OpenAI Gym-formatted action, 
  this function extracts a valid move for the game CoMaze, i.e.
  either a direction string alone, or a dictionnary containing 
  the keys 'action' (for the directional move, the value is a string)
  and 'symbol_Message' (for the message communicated to the other
  player, with the value being a string of uppercase symbol among:
  ['Q','W','E','R','T','Y','U','I','O','P']).
  """
  global DiscreteDirectionOnlyFormator
  
  move = DiscreteDirectionOnlyFormator.format_move(action)

  return move