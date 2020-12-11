def dummy_extract_exp_fn(game, player):
  """
  Provided a game state and player information, 
  this function extracts an experience tuple, i.e.
  (observation, reward, done, infos) in a typical RL fashion.
  """
  return game


def dummy_format_move_fn(action):
  """
  Provided an RL-based OpenAI Gym-formatted action, 
  this function extracts a valid move for the game CoMaze, i.e.
  either a direction string alone, or a dictionnary containing 
  the keys 'action' (for the directional move, the value is a string)
  and 'symbol_Message' (for the message communicated to the other
  player, with the value being a string of uppercase symbol among:
  ['Q','W','E','R','T','Y','U','I','O','P']).
  """
  return action