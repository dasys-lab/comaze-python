import os
import requests
import time
import gym
from gym.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete


from comaze.env import CoMaze 
from comaze.utils.gym_wrappers import CoMazeGymDictObsActionWrapper

class DictObsDiscreteActionExpExtractorClass(CoMazeGymDictObsActionWrapper):
  def __init__(self, vocab_size=10, maximum_sentence_length=1, options={}):
    # We cannot initialise it as a normal wrapper since we have no environment to wrap...
    self.nb_directions = 4
    self.actionId2action =  ["LEFT", "RIGHT", "UP", "DOWN"]
    self.action2actionId =  {"LEFT":0, "RIGHT":1, "UP":2, "DOWN":3}
  
    self.vocab_size = vocab_size
    assert self.vocab_size == 10
    self.token2id = {
      "empty":0, 
      "Q":1, 
      "W":2, 
      "E":3, 
      "R":4, 
      "T":5, 
      "Y":6, 
      "U":7, 
      "I":8, 
      "O":9, 
      "P":10
    }
    self.id2token = {
      0:"empty", 
      1:"Q", 
      2:"W", 
      3:"E", 
      4:"R", 
      5:"T", 
      6:"Y", 
      7:"U", 
      8:"I", 
      9:"O", 
      10:"P"
    }
    self.maximum_sentence_length = maximum_sentence_length
    assert self.maximum_sentence_length == 1
    self._build_sentenceId2sentence()
    
    # Action Space:
    self.nb_possible_actions = self.nb_directions*self.nb_possible_sentences+1 
    # +1 accounts for the SKIP action...
    
    # Observation Space:
    ## previous_message_space : 
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
    ##

  def extract_exp(self, game, player):
    obs = {}
    obs["encoded_pov"] = self._encode_game(game=game)
    
    obs["available_actions"] = self._get_available_actions(game=game)
    
    obs["previous_message"] = self._get_previous_message(game=game)
    
    if game["state"]["won"]:
      reward = 1
    elif game["state"]["lost"]:
      reward = -1
    else:
      reward = 0

    done = game["state"]["over"]

    infos = None

    return obs, reward, done, infos


DictObsDiscreteActionExpExtractor = DictObsDiscreteActionExpExtractorClass()
def extract_exp_fn(game, player):
  """
  Provided a game state and player information, 
  this function extracts an experience tuple, i.e.
  (observation, reward, done, infos) in a typical RL fashion.
  """
  global DictObsDiscreteActionExpExtractor
  return DictObsDiscreteActionExpExtractor.extract_exp(game, player)


class DiscreteActionMoveExtractorClass(object):
  def __init__(self, vocab_size=10, maximum_sentence_length=1, options={}):
    self.nb_directions = 4
    self.vocab_size = vocab_size
    assert self.vocab_size == 10
    self.maximum_sentence_length = maximum_sentence_length
    assert self.maximum_sentence_length == 1
    
    self.nb_possible_sentences = 1 # account for the empty string:
    for pos in range(self.maximum_sentence_length):
      self.nb_possible_sentences += (self.vocab_size)**(pos+1)
    
    # Action Space:
    self.nb_possible_actions = self.nb_directions*self.nb_possible_sentences+1 
    # +1 accounts for the SKIP action...
    self.action_space = Discrete(self.nb_possible_actions)

  def extract_move(self, action):
    if not self.action_space.contains(action):
      raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

    if action != (self.nb_possible_actions-1):
      original_action_direction_id = action // self.nb_possible_sentences
      original_action_direction = self.actionId2action[original_action_direction_id]
      
      original_action_sentence_id = (action % self.nb_possible_sentences)
      original_action_sentence = self.sentenceId2sentence[original_action_sentence_id]
      original_action_message = self._get_message_from_sentence(original_action_sentence)
    else:
      original_action_direction = "SKIP"
      original_action_message = None #self.sentenceId2sentence[0] #empty message.
    
    if original_action_direction != "SKIP":
      return original_action_direction

    rd = {
      'direction':original_action_direction,
      'symbol_Message':original_action_message
    }

    return rd 

DiscreteActionMoveExtractor = DiscreteActionMoveExtractorClass()
def extract_move_fn(action):
  """
  Provided an RL-based OpenAI Gym-formatted action, 
  this function extracts a valid move for the game CoMaze, i.e.
  either a direction string alone, or a dictionnary containing 
  the keys 'action' (for the directional move, the value is a string)
  and 'symbol_Message' (for the message communicated to the other
  player, with the value being a string of uppercase symbol among:
  ['Q','W','E','R','T','Y','U','I','O','P']).
  """
  global DiscreteActionMoveExtractor
  
  move = DiscreteActionMoveExtractor.extract_move(action)

  return move

class RLCoMaze(CoMaze):
  def __init__(self, extract_exp_fn=extract_exp_fn, extract_move_fn=extract_move_fn):
    super(RLCoMaze, self).__init__()
    if os.path.isfile(".local"):
      self.API_URL = "http://localhost:16216"
      self.WEBAPP_URL = "http://localhost"
    else:
      self.API_URL = "http://teamwork.vs.uni-kassel.de:16216"
      self.WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
    self.LIB_VERSION = "1.3.0"

    self.extract_exp_fn = extract_exp_fn
    self.extract_move_fn = extract_move_fn

    self.build_agent()


  def build_agent(self):
    raise NotImplementedError

  def next_move(self, game, player):
    obs, reward, done, infos = self.extract_exp_fn(game, player)

    next_action = self.forward(obs=obs, reward=reward, done=done, infos=infos)

    next_move = self.extract_move_fn(next_action)

    return next_move

  def forward(self, obs, reward=0, done=False, infos=None):
    raise NotImplementedError