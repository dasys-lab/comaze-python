import gym
from gym.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete
import numpy as np

class CoMazeGymActionWrapper(gym.Wrapper):
  def __init__(self, env, vocab_size=10, maximum_sentence_length=1, options={}):
    super(CoMazeGymActionWrapper, self).__init__(env)
    self.nb_directions = 4
    self.actionId2action =  ["LEFT", "RIGHT", "UP", "DOWN"]
    
    self.vocab_size = vocab_size
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
    self._build_sentenceId2sentence()
    
    self.nb_possible_actions = self.nb_directions*self.nb_possible_sentences+1 
    # +1 accounts for the SKIP action...
    self.action_space = Discrete(self.nb_possible_actions)

  def _build_sentenceId2sentence(self):
    self.nb_possible_sentences = 1 # account for the empty string:
    for pos in range(self.maximum_sentence_length):
      self.nb_possible_sentences += (self.vocab_size)**(pos+1)
    sentenceId2sentence = np.zeros( (self.nb_possible_sentences, self.maximum_sentence_length))
    idx = 1
    local_token_pointer = 0
    global_token_pointer = 0
    while idx != self.nb_possible_sentences:
      sentenceId2sentence[idx] = sentenceId2sentence[idx-1]
      sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
      
      while sentenceId2sentence[idx][local_token_pointer] == 0:
        # remove the possibility of an empty symbol on the left of actual tokens:
        sentenceId2sentence[idx][local_token_pointer] += 1
        local_token_pointer += 1
        sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
      idx += 1
      local_token_pointer = 0
    
    self.sentenceId2sentence = sentenceId2sentence
  
  def _get_message_from_sentence(self, sentence):
    message = []
    for pos, sidx in enumerate(sentence):
        # if empty symbol, then there is nothing on the right of it:
        if sidx == 0: 
          # if empty sentence:
          if pos == 0:
            message = None
          break
        token = self.id2token[sidx]
        message.append(token)
    
    return message

  def step(self, action):
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
    
    print(f'discrete action {action} -> original action: direction={original_action_direction} / message={original_action_message}')
    
    return self.env.step(action=original_action_direction, message=original_action_message)



class CoMazeGymDictObsActionWrapper(gym.Wrapper):
    """
    
    """
    def __init__(self, env, vocab_size=10, maximum_sentence_length=1, options={}):
        super(CoMazeGymDictObsActionWrapper, self).__init__(env)
        self.game = self.env.reset()
        
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
        self.action_space = Discrete(self.nb_possible_actions)
        
        # Observation Space:
        ## previous_message_space
        previous_message_space = MultiDiscrete(
            [self.vocab_size+1 for _ in range(self.maximum_sentence_length)]
        )
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
        
        encoded_pov_space = Box(
            low=0, 
            high=1, 
            shape=(
                self.game["config"]["arenaSize"]["x"],
                self.game["config"]["arenaSize"]["y"],
                self.nb_different_tile
            ),
            dtype=np.int64, 
        )
        
        ## available_action_space:
        available_actions_space = MultiBinary(n=self.nb_possible_actions)
        
        ##
        
        self.observation_space = Dict({
          'encoded_pov': encoded_pov_space,
          'available_actions': available_actions_space,
          'previous_message': previous_message_space,
        })

    def _build_sentenceId2sentence(self):
        self.nb_possible_sentences = 1 # account for the empty string:
        for pos in range(self.maximum_sentence_length):
            self.nb_possible_sentences += (self.vocab_size)**(pos+1)
        sentenceId2sentence = np.zeros( (self.nb_possible_sentences, self.maximum_sentence_length))
        idx = 1
        local_token_pointer = 0
        global_token_pointer = 0
        while idx != self.nb_possible_sentences:
            sentenceId2sentence[idx] = sentenceId2sentence[idx-1]
            sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
            
            while sentenceId2sentence[idx][local_token_pointer] == 0:
                # remove the possibility of an empty symbol on the left of actual tokens:
                sentenceId2sentence[idx][local_token_pointer] += 1
                local_token_pointer += 1
                sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
            idx += 1
            local_token_pointer = 0    
        
        self.sentenceId2sentence = sentenceId2sentence
  
    def _get_message_from_sentence(self, sentence):
        message = []
        for pos, sidx in enumerate(sentence):
            # if empty symbol, then there is nothing on the right of it:
            if sidx == 0: 
                # if empty sentence:
                if pos == 0:
                    message = None
                break
            token = self.id2token[sidx]
            message.append(token)
    
        return message
    
    def reset(self, options={}):
        level = options.get("level", "1")
        num_of_player_slots = options.get("num_of_player_slots", "2")
        self.game_id = requests.post(self.API_URL + "/game/create?level=" + level + "&numOfPlayerSlots=" + num_of_player_slots).json()["uuid"]
        options["game_id"] = self.game_id
        
        self.game = self.play_existing_game(options)
        
        self.obs = {}
        self.obs["encoded_pov"] = self._encode_game(game=self.game)
        
        self.obs["available_actions"] = self._get_available_actions(game=self.game)
        
        self.obs["previous_message"] = np.zeros(self.maximum_sentence_length, dtype=np.int64) #self._get_previous_message(game=self.game)
        
        return self.obs
    
    def step(self, action):
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
            
        print(f'discrete action {action} -> original action: direction={original_action_direction} / message={original_action_message}')

        self.game, self.reward, self.done, self.infos = self.env.step(action=original_action_direction, message=original_action_message)
        
        self.obs = {}
        self.obs["encoded_pov"] = self._encode_game(game=self.game)
        
        self.obs["available_actions"] = self._get_available_actions(game=self.game)
        
        self.obs["previous_message"] = self._get_previous_message(game=self.game)
        import ipdb; ipdb.set_trace()
        
        return self.obs, self.reward, self.done, self.infos
        
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
            if goal in unreached_goals:
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
    
    def _get_available_actions(self, game):
        current_player_available_actions = game["currentPlayer"]["actions"]
        a_actions = np.zeros(self.nb_possible_actions)
        # SKIP action:
        a_actions[-1] = 1
        for action in current_player_available_actions:
            if action == "SKIP":    continue
            action_id = self.action2actionId[action]
            for midx in range(self.nb_possible_sentences):
                a_actions[action_id*self.nb_possible_sentences+midx] = 1
        return a_actions
    
    def _get_previous_message(self, game):
        players = game["players"]
        currentPlayer = game["currentPlayer"]
        otherPlayers = [player for player in players if player != currentPlayer]
        assert len(otherPlayers) == 1
        otherPlayer_message = otherPlayer[0]["lastSymbolMessage"]
        otherPlayer_message_discrete = np.zeros(self.maximum_sentence_length)
        for widx, token in zip(range(self.maximum_sentence_length), otherPlayer_message):
            otherPlayer_message_discrete[widx] = self.token2id[token]
        return self._get_message_from_sentence(sentence=otherPlayer_message_discrete)
    