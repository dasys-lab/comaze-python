import os
import requests
import time

from .logging_abstract_env import LoggingAbstractEnv 

class CoMaze(LoggingAbstractEnv):
  if os.path.isfile(".local"):
    API_URL = "http://localhost:16216"
    WEBAPP_URL = "http://localhost"
  else:
    API_URL = "http://teamwork.vs.uni-kassel.de:16216"
    WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  LIB_VERSION = "1.3.0"
  
  def __init__(self, logging_base_path="./logs/games/"):
    super(CoMaze, self).__init__(logging_base_path=logging_base_path)
    self.game_id = None 
    self.player_id = None

  def next_move(self, game, player):
    return ":("

  def play_new_game(self, options={}):
    level = options.get("level", "1")
    num_of_player_slots = options.get("num_of_player_slots", "2")
    self.game_id = requests.post(self.API_URL + "/game/create?level=" + level + "&numOfPlayerSlots=" + num_of_player_slots).json()["uuid"]
    options["game_id"] = self.game_id
    self.play_existing_game(options)

  def play_existing_game(self, options={}):
    if "look_for_player_name" in options:
      options["game_id"] = requests.get(self.API_URL + "/game/byPlayerName?playerName=" + options["look_for_player_name"]).json()["uuid"]

    if "game_id" not in options or len(options["game_id"]) != 36:
      raise Exception("You must provide a game id when attending an existing game. Use play_new_game() instead of play_existing_game() if you want to create a new game.")

    player_name = options.get("player_name", "Python")
    self.game_id = options["game_id"]
    player = requests.post(self.API_URL + "/game/" + self.game_id + "/attend?playerName=" + player_name).json()
    self.player_id = player["uuid"]

    print("Joined gameId: " + self.game_id)
    print("Playing as playerId: " + self.player_id)

    self._init_logger()

    self.game_loop(self.game_id, player)

  def game_loop(self, game_id, player):
    game = requests.get(self.API_URL + "/game/" + game_id).json()

    while not game["state"]["over"]:
      game = requests.get(self.API_URL + "/game/" + game_id).json()

      if not game["state"]["started"]:
        print("Waiting for players. (Invite someone: " + self.WEBAPP_URL + "/?gameId=" + game_id + ")")
        time.sleep(3)
        continue

      if game["currentPlayer"]["uuid"] != player["uuid"]:
        print("Not my turn. Waiting.")
        time.sleep(1)
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

      print("Moving " + action)
      request_url = self.API_URL + "/game/" + game_id + "/move"
      request_url += "?playerId=" + player["uuid"]
      request_url += "&action=" + action
      if symbol_message:
        request_url += "&symbolMessage=" + symbol_message
      requests.post(request_url)

      if game["state"]["won"]:
        print("Game won!")
        reward = 1
      elif game["state"]["lost"]:
        print("Game lost (" + game["state"]["lostMessage"] + ").")
        reward = -1
      else:
        reward = 0

      self._log(action=action, message=symbol_message, reward=reward)


class CoMazeGym(LoggingAbstractEnv):
  if os.path.isfile(".local"):
    API_URL = "http://localhost:16216"
    WEBAPP_URL = "http://localhost"
  else:
    API_URL = "http://teamwork.vs.uni-kassel.de:16216"
    WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  LIB_VERSION = "1.1.0"
  
  def __init__(self, logging_base_path="./logs/games/"):
    super(CoMazeGym, self).__init__(logging_base_path=logging_base_path)
    self.game = None
    self.game_id = None
    self.player_id = None
    self.action_space = None
    
  def reset(self, options={}):
    level = options.get("level", "1")
    num_of_player_slots = options.get("num_of_player_slots", "2")
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
    while self.game['currentPlayer']['uuid'] != self.player_id:
      print(f'Waiting for other player to make first move')
      time.sleep(1)
      self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()
    
    self._init_logger()
    
    return self.game

  def step(self, action, message=None):
    moved = False
    while not moved:
      self.game = requests.get(self.API_URL + "/game/" + self.game_id).json()

      if not self.game["state"]["started"]:
        print("Waiting for players. (Invite someone: " + self.WEBAPP_URL + "/?gameId=" + self.game_id + ")")
        time.sleep(3)
        continue

      print("Moving " + action)
      print(f'Sending message {message}')
      print('---')
      self.game = requests.post(self.API_URL + "/game/" + self.game_id + "/move?playerId=" + self.player_id + "&action=" + action).json()
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
    
    self._log(action=action, message=message, reward=reward)
    
    return self.game, reward, self.game["state"]["over"], None
    
