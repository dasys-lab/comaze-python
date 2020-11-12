import requests
import random

exec(requests.get('http://teamwork.vs.uni-kassel.de/comaze.py').text)
exec(open("comaze.py").read())


class CustomCoMaze(CoMaze):

  def next_move(self, game, player):
    return random.choice(player["actions"])


CustomCoMaze().play_new_game({
  "level": "1",
  "num_of_player_slots": "1"
})
