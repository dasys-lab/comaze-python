import requests
import random

exec(requests.get('http://teamwork.vs.uni-kassel.de/comaze.py').text)
exec(open("comaze.py").read())  # for local development. remove this line in colab notebook.


class CustomCoMaze(CoMaze):

  def next_move(self, game, player):
    # This method is called whenever it is your turn

    print("Hi, my name is " + player["name"] + ".")
    print("I can move " + player["actions"][0] + ", for example.")
    print("We have used " + str(game["usedMoves"]) + " moves so far.")
    print("There are " + str(len(game["unreachedGoals"])) + " goals left to reach.")

    # Use the information contained in the dicts "game" and "player" to determine your next move

    action = random.choice(player["actions"])
    symbol_message = random.choice(game["config"]["symbolMessages"])  # optional

    return {
      "action": action,
      "symbol_message": symbol_message  # optional
    }

# Example 1: Start a new game
# CustomCoMaze().play_new_game({
#   "level": "1",
#   "num_of_player_slots": "1",
#   "player_name": "Python Example 1"
# })

# Example 2: Attend a game using its UUID
# CustomCoMaze().play_existing_game({
#   "game_id": "2d4e4d82-6417-489f-b9b1-ed4c0b801b6f",
#   "player_name": "Python Example 2"
# })

# Example 3: Attend a game where Alice is waiting for players
# CustomCoMaze().play_existing_game({
#   "look_for_player_name": "Alice",
#   "player_name": "Python Example 3"
# })
