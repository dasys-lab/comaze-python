import gym
from gym.spaces import Box, Discrete, Dict, MultiBinary
import numpy as np

class CoMazeGymActionWrapper(gym.Wrapper):
  def __init__(self, env, vocab_size=10, maximum_sentence_length=1, options={}):
    super(CoMazeGymActionWrapper, self).__init__(env)
    self.nb_directions = 4
    self.actionId2action =  ["LEFT", "RIGHT", "UP", "DOWN" ]
    
    self.vocab_size = vocab_size
    self.maximum_sentence_length = maximum_sentence_length
    self._build_sentenceId2sentence()
    
    self.nb_possible_actions = self.nb_directions*self.nb_possible_sentences
    self.action_space = Discrete(self.nb_possible_actions)

  def _build_sentenceId2sentence(self):
    self.nb_possible_sentences = (self.vocab_size+1)**self.maximum_sentence_length
    sentenceId2sentence = np.zeros( (self.nb_possible_sentences, self.maximum_sentence_length))
    
    idx = 1
    local_token_pointer = 0
    global_token_pointer = 0
    while idx != self.nb_possible_sentences:
      sentenceId2sentence[idx] = sentenceId2sentence[idx-1]
      sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
      while sentenceId2sentence[idx][local_token_pointer] == 0:
        local_token_pointer += 1
        sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
      idx += 1
      local_token_pointer = 0
    
    self.sentenceId2sentence = sentenceId2sentence
  
  def step(self, action):
    if not self.action_space.contains(action):
      raise ValueError('action {} is invalid for {}'.format(action, self.action_space))
    
    original_action_direction_id = action // self.nb_possible_sentences
    original_action_direction = self.actionId2action[original_action_direction_id]
    
    original_action_message_id = (action % self.nb_possible_sentences)
    original_action_message = self.sentenceId2sentence[original_action_message_id]

    print(f'discrete action {action} -> original action: direction={original_action_direction} / message={original_action_message}')
    
    return self.env.step(action=original_action_direction, message=original_action_message)