import os
import requests
import time

import numpy as np
from tensorboardX import SummaryWriter

class LoggingAbstractEnv:
  def __init__(self, logging_base_path="./logs/games/"):
    self.action2actionId =  {"LEFT":0, "RIGHT":1, "UP":2, "DOWN":3, "SKIP":4}
    self.token2tokenId = {
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
    
    self.logging_base_path = logging_base_path
    self.per_episode_logging_iteration = 0
    self.logger = None
    
  def _init_logger(self):
    """
    Assumes that the inherited class has attribute game_id and player_id.
    """
    self.logging_folder_path = os.path.join(self.logging_base_path, f"{self.game_id}")
    os.makedirs(self.logging_folder_path, exist_ok=True)
    self.logging_log_path = os.path.join(self.logging_folder_path, f"{self.player_id}.tlog")
    self.logger = SummaryWriter(self.logging_log_path)
    self.per_episode_logging_iteration = 0
    
  def _log(self, action, message, reward):
    if message is None:
        message = "empty"
    
    self.logger.add_scalar("Training/Reward", reward, self.per_episode_logging_iteration)
    
    action_id = self.action2actionId[action]
    message_id = self.token2tokenId[message]
    
    print(f"Logging: action::{action_id} / message::{message_id}." )
    
    self.logger.add_scalar("Training/Action", action_id, self.per_episode_logging_iteration)
    self.logger.add_scalar("Training/Message", message_id, self.per_episode_logging_iteration)
    
    action_hist =  np.zeros(len(self.action2actionId), dtype=np.int64)
    action_hist[action_id] = 1
    message_hist =  np.zeros(len(self.token2tokenId), dtype=np.int64)
    message_hist[message_id] = 1
    
    self.logger.add_histogram("Traning/ActionHist", action_hist, self.per_episode_logging_iteration)
    self.logger.add_histogram("Traning/MessageHist", message_hist, self.per_episode_logging_iteration)
    
    self.per_episode_logging_iteration += 1
    self.logger.flush()