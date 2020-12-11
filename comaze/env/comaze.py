import abc
import copy
import os
import random
import time
import requests
from typing import Any
from typing import Callable
from typing import Dict, List
from typing import Optional
from typing import Tuple

import gym


class CoMaze:
  if os.path.isfile(".local"):
    API_URL = "http://localhost:16216"
    WEBAPP_URL = "http://localhost"
  else:
    API_URL = "http://teamwork.vs.uni-kassel.de:16216"
    WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  LIB_VERSION = "1.3.0"

  def next_move(self, game, player):
    raise NotImplementedError

  def play_new_game(self, options={}):
    level = options.get("level", "1")
    num_of_player_slots = options.get("num_of_player_slots", "2")
    game_id = requests.post(self.API_URL + "/game/create?level=" + level + "&numOfPlayerSlots=" + num_of_player_slots).json()["uuid"]
    options["game_id"] = game_id
    self.play_existing_game(options)

  def play_existing_game(self, options={}):
    print("playing existing game", options)
    if "look_for_player_name" in options:
      options["game_id"] = requests.get(self.API_URL + "/game/byPlayerName?playerName=" + options["look_for_player_name"]).json()["uuid"]

    if "game_id" not in options or len(options["game_id"]) != 36:
      raise Exception("You must provide a game id when attending an existing game. Use play_new_game() instead of play_existing_game() if you want to create a new game.")

    player_name = options.get("player_name", "Python")
    game_id = options["game_id"]
    player = requests.post(self.API_URL + "/game/" + game_id + "/attend?playerName=" + player_name).json()
    print("Joined gameId: " + game_id)
    print("Playing as playerId: " + player["uuid"])
    self.game_loop(game_id, player)

  def game_loop(self, game_id, player):
    game = requests.get(self.API_URL + "/game/" + game_id).json()

    while not game["state"]["over"]:
      game = requests.get(self.API_URL + "/game/" + game_id).json()

      if not game["state"]["started"]:
        print("Waiting for players. (Invite someone: " + self.WEBAPP_URL + "/?gameId=" + game_id + ")")
        time.sleep(3)
        continue

      if game["currentPlayer"]["uuid"] != player["uuid"]:
        print(f"Not my turn. Waiting. (should be {game['currentPlayer']['uuid']}, but I am {player['uuid']}")
        print("We have used " + str(game["usedMoves"]) + " moves so far.")
        time.sleep(0.5)
        continue

      next_move = self.next_move(game, player)

      # next_move can be a direction/skip string (maintaining compatibility with APIs <= 1.1) or a dict containing direction and an optional symbolMessage
      action = None
      symbol_message = None
      if type(next_move) == str:
        action = next_move
      elif type(next_move) == dict:
        action = next_move.get("action")
        symbol_message = next_move.get("symbol_message")

      print("Moving " + str(action))
      request_url = self.API_URL + "/game/" + game_id + "/move"
      request_url += "?playerId=" + player["uuid"]
      request_url += "&action=" + action
      if symbol_message:
        request_url += "&symbolMessage=" + symbol_message
      requests.post(request_url)

    if game["state"]["won"]:
      print("Game won!")
    elif game["state"]["lost"]:
      print("Game lost (" + game["state"]["lostMessage"] + ").")


class CoMazeGym(gym.Env):
  if os.path.isfile(".local"):
    API_URL = "http://localhost:16216"
    WEBAPP_URL = "http://localhost"
  else:
    API_URL = "http://teamwork.vs.uni-kassel.de:16216"
    WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  LIB_VERSION = "1.3.0"
  
  def __init__(self):
    self.game = None
    self.game_id = None
    self.player_id = None
    self.action_space = None

  def reset(self, options={}):
    level = options.get("level", "1")
    num_of_player_slots = options.get("num_of_player_slots", "2")
    
    self.game_id = options.get("game_id", None)
    if self.game_id is None:
      self.game_id = requests.post(self.API_URL + "/game/create?level=" + level + "&numOfPlayerSlots=" + num_of_player_slots).json()["uuid"]
      options["game_id"] = self.game_id
    
    return self.play_existing_game(options)

  def play_existing_game(self, options={}):
    if "look_for_player_name" in options:
      options["game_id"] = requests.get(self.API_URL + "/game/byPlayerName?playerName=" + options["look_for_player_name"]).json()["uuid"]

    if "game_id" not in options or len(options["game_id"]) != 36:
      raise Exception("You must provide a game id when attending an existing game. Use play_new_game() instead of play_existing_game() if you want to create a new game.")

    player_name = options.get("player_name", "Python")
    self.game_id = options["game_id"]
    print("Joined gameId: " + self.game_id)
    player = requests.post(self.API_URL + "/game/" + self.game_id + "/attend?playerName=" + player_name).json()
    self.player_id = player["uuid"]
    self.action_space = player['directions'] + ['SKIP']
    print("Playing as playerId: " + self.player_id)
    self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()
    print(f'Action Space is {self.action_space}')

    while self.game['currentPlayer']['uuid'] != self.player_id or len(self.game["players"]) < 2:
      if self.game['currentPlayer']['uuid'] != self.player_id:
        print(f'Waiting for other player to make first move')
      print("(Invite someone: " + self.WEBAPP_URL + "/?gameId=" + self.game_id + " )")
      time.sleep(1)
      self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()

    return self.game

  def step(self, action, message=None):
    moved = False
    while not moved:
      self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()

      if not self.game["state"]["started"]:
        print("Waiting for players. (Invite someone: " + self.WEBAPP_URL + "/?gameId=" + self.game_id + " )")
        time.sleep(3)
        continue
      
      available_actions = self.game["currentPlayer"]["directions"]+["SKIP"]
      if action not in available_actions:
        print(f"WARNING: Action {action} is not available to the current player.")
        action = "SKIP"
      print("Moving " + action)
      if action == "SKIP":
        print(f'Wanted to send message {message}, but skipped.')
        message = None
      else:
        print(f'Sending message {message}.')
      print('---')
      request_url = self.API_URL + "/game/" + self.game_id + "/move"
      request_url += "?playerId=" + self.player_id
      request_url += "&action=" + action
      if message is not None and action != 'SKIP':
        request_url += "&symbolMessage=" + message
      print(request_url)
      self.game = requests.post(request_url).json()
      moved = True
    
    if self.game["state"]["won"]:
      print("Game won!")
      reward = 1
    elif self.game["state"]["lost"]:
      print("Game lost (" + self.game["state"]["lostMessage"] + ").")
      reward = -1
    else:
      reward = 0

    if not self.game["state"]["over"]:
      # wait for other player to make a move before sending back obs
      while self.game['currentPlayer']['uuid'] != self.player_id:
        print(f'Waiting for other player to make a move')
        time.sleep(1)
        self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()

    return self.game, reward, self.game["state"]["over"], None


# Type definitions.
Observation = Dict[str, Any]
Action = Dict[str, str]


class TwoPlayersCoMazeGym(gym.Env):
  """
  OpenAI gym environment for the 2-players CoMaze game.
  """
  if os.path.isfile(".local"):
    _API_URL = "http://localhost:16216"
    _WEBAPP_URL = "http://localhost"
  else:
    _API_URL = "http://teamwork.vs.uni-kassel.de:16216"
    _WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  _LIB_VERSION = "1.3.0"

  def __init__(
    self, 
    level: Optional[str]=None, 
    verbose: Optional[Any]=False, 
    sparse_reward: Optional[Any]=False,
    ) -> None:
    """
    Initializes an environment.
    """
    self._agent_ids = [None, None]
    self._action_spaces = [None, None]
    self._level = str(level or "1")
    self._game_id = None
    self._time_index = -1
    self.game = None
    self.verbose = verbose
    self._sparse_reward = sparse_reward
    self.reached_goals = 0
  
  @property
  def action_space(self) -> List[gym.spaces.Space]:
    """
    Returns the agents' action space.
    """
    return [copy.deepcopy(space) for space in self._action_spaces]

  def reset(self) -> Observation:
    """
    Resets simulation and generates a new game.
    """
    self._agent_ids = [None, None]
    self._action_spaces = [None, None]
    self._game_id = requests.post(
        self._API_URL + "/game/create?level=" + self._level + "&numOfPlayerSlots=2"
    ).json()["uuid"]
    for i in range(2): # We assume two-players games only.
      player = requests.post(
        self._API_URL + "/game/" + self._game_id + "/attend?playerName=agent_".format(i)).json()
      self._agent_ids[i] = player["uuid"]
      self._action_spaces[i] = player["directions"] + ["SKIP"]
    self._time_index = 0
    
    return requests.get(self._API_URL + "/game/" + self._game_id).json()
  
  def step(self, action: Dict[str, str]) -> Tuple[Observation, float, bool, Any]:
    """
    Performs a single step in the environment.
    """
    #assert action in self.action_space[self._time_index%2]
    message = action.get("symbol_Message", None)
    action = action.get("direction", "SKIP")

    # Fetch game json:
    if self.verbose:  print('---')
    fetch_url = self._API_URL + "/game/" + self._game_id
    self.game = requests.get(fetch_url).json()
    ## game state:
    game_state = self.game["state"]
    assert game_state["started"]
    if self.verbose:  print(self._time_index, game_state)

    # Apply action:
    ## Verify action is compatible:
    available_actions = self.game["currentPlayer"]["directions"]+["SKIP"]
    if action not in available_actions:
      if self.verbose:  print(f"WARNING: Action {action} is not available to the current player.")
      action = "SKIP"
    if self.verbose:  print("Moving " + action)
    if action == "SKIP":
      if self.verbose:  print(f'Wanted to send message {message}, but skipped.')
      message = None
    else:
      if self.verbose:  
        print(f'Sending message {message}.')
    
    request_url = self._API_URL + "/game/" + self._game_id + "/move"
    request_url += "?playerId=" + self._agent_ids[self._time_index%2] #self.player_id
    request_url += "&action=" + action
    if message is not None and action != 'SKIP':
      request_url += "&symbolMessage=" + message
    if self.verbose:  print(request_url)
    try:
      self.game = requests.post(request_url).json()
    except Exception as e:
      if self.verbose:  
        #The agent is likely trying to move outside the arena.
        print(f"WARNING: {e.doc}")
      # Regularising the resulting game state:
      self.game = requests.get(fetch_url).json()
      #time.sleep(1)
    resulting_game_state = self.game["state"]
    if self.verbose:  print('---')
    
    # Book-keeping.
    self._time_index = self._time_index + 1

    # Calculate reward.
    reward = 0
    if self._sparse_reward:
      if resulting_game_state ["won"]:
        reward = 1
      elif resulting_game_state ["lost"]:
        if self.verbose:  print("Game lost (" + resulting_game_state["lostMessage"] + ").")
        reward = -1
    else:
      reached_goals = 4-len(self.game["unreachedGoals"])
      if reached_goals != self.reached_goals:
        self.reached_goals = reached_goals
        reward = 1
      else:
        reward = -0.1

      if resulting_game_state ["won"]:
        reward += 4
      elif resulting_game_state ["lost"]:
        if self.verbose:  print("Game lost (" + resulting_game_state["lostMessage"] + ").")
        reward -= 4

    return copy.deepcopy(self.game), reward, resulting_game_state["over"], None