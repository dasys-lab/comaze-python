import requests
import random

exec(requests.get('http://teamwork.vs.uni-kassel.de/comaze.py').text)
exec(open("comaze.py").read())


class CustomCoMaze(CoMaze):

  def next_move(self, game, player):
    return random.choice(player["actions"])

# Example 1: Start a new game
# CustomCoMaze().play_new_game({
#   "level": "1",
#   "num_of_player_slots": "1"
# })

# Example 2: Attend a game using its UUID
# CustomCoMaze().play_existing_game({
#   "game_id": "2d4e4d82-6417-489f-b9b1-ed4c0b801b6f"
# })

# Example 3: Attend a game looking for another player's name
# CustomCoMaze().play_existing_game({
#   "look_for_player_name": "Alice"
# })
