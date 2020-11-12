import requests
import time


class CoMaze:
  API_URL = "http://teamwork.vs.uni-kassel.de:16216"
  WEBAPP_URL = "http://teamwork.vs.uni-kassel.de"
  LIB_VERSION = "0.0.1"

  def next_move(self, game, player):
    return ":("

  def play_new_game(self, options=None):
    if options is None:
      options = {}
    level = options.get("level", "1")
    num_of_player_slots = options.get("num_of_player_slots", "2")
    player_name = options.get("player_name", "Python")
    game_id = requests.post(self.API_URL + "/game/create?level=" + level + "&numOfPlayerSlots=" + num_of_player_slots).json()["uuid"]
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
        print("Not my turn. Waiting.")
        time.sleep(1)
        continue

      direction = self.next_move(game, player)
      print("Moving " + direction)
      requests.post(self.API_URL + "/game/" + game_id + "/move?playerId=" + player["uuid"] + "&action=" + direction)

    if game["state"]["won"]:
      print("Game won!")
    elif game["state"]["lost"]:
      print("Game lost (" + game["state"]["lostMessage"] + ").")
