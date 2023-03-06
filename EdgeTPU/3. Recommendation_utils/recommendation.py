#import tensorflow as tf
#from tensorflow import keras
import math
import numpy as np
import json
from pycoral.adapters.common import output_tensor
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

class RecommenderModel():
    def __init__(self, model_path):        
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.closet = dict()        
        self.closet2img = []
        self.closet_load()

    def closet_load(self):
        with open('closet/closet.json', 'r') as f:
          json_file = json.load(f)
        self.closet["top"] = json_file["top"]
        self.closet["bot"] = json_file["bot"]
        
        with open('closet/idx2img.json', 'r') as f:
          j = json.load(f)
        self.closet2img = j
            
    def evaluate_score(self, vector200, label_idx):
      # Load the TFLite model and allocate tensors.
      #vec_array = np.array(vector200, dtype=np.uint8)
      run_inference(self.interpreter, vector200.tobytes())
      vector4 = output_tensor(self.interpreter, 0)
      #[[0.1, 0.2, 0.3, 0.4]]
      score = vector4[0, label_idx]
      return score

    def topRecommend(self, bot_embed, label_idx):
        closet_top = self.closet["top"]
        max_ret = (0, None)
        for top in closet_top:
          idx = top["idx"]
          top_embed = top["embed"]
          top_embed = np.array([int(x) for x in top_embed], dtype = np.uint8)
          input = np.concatenate((bot_embed, top_embed), axis=0)
          score = self.evaluate_score(input, label_idx)
          if(score > max_ret[0]):
            max_ret = (score, idx)
        return max_ret

    def botRecommend(self, top_embed, label_idx):
        closet_bot = self.closet["bot"]
        max_ret = (0, None)
        for bot in closet_bot:
          idx = bot["idx"]
          bot_embed = bot["embed"]
          bot_embed = np.array([int(x) for x in bot_embed], dtype = np.uint8)
          input = np.concatenate((top_embed, bot_embed), axis=0)
          score = self.evaluate_score(input, label_idx)
          if(score > max_ret[0]):
            max_ret = (score, idx)
        return max_ret