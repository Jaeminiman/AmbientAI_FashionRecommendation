import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

class RecommenderModel():
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Flatten(input_shape = (200,)))                
        self.model.add(keras.layers.Dense(256,activation = 'relu'))                  
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(128,activation = 'relu'))                          
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(64,activation = 'relu'))         
        self.model.add(keras.layers.Dropout(0.3))                            
        self.model.add(keras.layers.Dense(4,activation = "softmax"))        
        
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer = keras.optimizers.Adam(lr = 0.0005,decay = 1e-6),
              metrics=["accuracy"])
    def set_testset(self, test_dataset):
      self.test_set = test_dataset      
            
    def set_trainset(self, train_dataset):
      self.train_set = train_dataset    
      print(len(train_dataset[1]))
      labels = [0,0,0,0]
      for t in train_dataset[1]:        
        if(t[0] == 1):
          labels[0] += 1
        elif(t[1]==1):
          labels[1] += 1
        elif(t[2] == 1):
          labels[2] +=1 
        elif(t[3] == 1):
          labels[3] +=1
      print(f" military : {labels[0]}, minimal : {labels[1]}, formal : {labels[2]}, street : {labels[3]}")
  
    def fit(self, epochs = 100, workers = 4):        
        self.model.fit(x = self.train_set[0], y= self.train_set[1], validation_split = 0.3 ,epochs = epochs, workers = workers, shuffle = True)
    
    def evaluate(self):
        return self.model.evaluate(self.test_set[0],self.test_set[1])
        
    def predict(self, input):                
        return self.model.predict(input)
    
      
class RecommenderModelController():
    def __init__(self, model):
        self.model = model
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
    
    def evaluate(self):
        return self.model.evaluate(self.test_set[0],self.test_set[1])
        
    def predict(self, input):                
        return self.model.predict(input)
    
    def topRecommend(self, bot_embed):
        closet_top = self.closet["top"]
        max_ret = (0, None)
        for top in closet_top:
          idx = top["idx"]
          top_embed = top["embed"]
          input = np.concatenate(top_embed, bot_embed)
          score = self.model.predict(input)
          if(score > max_ret.first):
            max_ret.first = score
            max_ret.second = idx                    
        return max_ret
    def botRecommend(self, top_embed):
        closet_bot = self.closet["bot"]
        max_ret = (0, None)
        for bot in closet_bot:
          idx = bot["idx"]
          bot_embed = bot["embed"]
          input = np.concatenate(top_embed, bot_embed)
          score = self.model.predict(input)
          if(score > max_ret.first):
            max_ret.first = score
            max_ret.second = idx                    
        return max_ret


