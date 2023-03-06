"""
Run train model:
python3 train.py \
  --extract true(or false) 
"""

import numpy as np
from recommendation import *
import tensorflow_datasets as tfds # for custom dataset
import argparse
from train_extract_embedding import extract_embedding


def main():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', help='if needed image to vector, type \'true\'',
                        default="False") 
    args = parser.parse_args()
    if(str.lower(args.extract) == "true"):
        extract_embedding()
    
        
    # Recommender Model load
    recModel = RecommenderModel()                
        
    ds, info = tfds.load('new_dataset',with_info = True)
    
    train_size = info.splits['train'].num_examples
    test_size = info.splits['test'].num_examples
    print(f"train_size : {train_size}, test_size : {test_size}")
    
    
    #(train_dataset, val_dataset) = tfds.load('new_dataset', split = ['train[:80%]', 'train[80%:]'], shuffle_files=True)
    train_dataset = tfds.load('new_dataset', split = tfds.Split.TRAIN, shuffle_files=True)    
    x = {"train" : [], "val" : [], "test" : []}
    y = {"train" : [], "val" : [], "test" : []}    
    print("train set")
    for data in train_dataset:
        emb, label = data['embedding'], data['label']        
        one_hot_label = np.zeros(4)
        one_hot_label[label] = 1
        x["train"].append(emb.numpy())
        y["train"].append(one_hot_label)            
    test_dataset = tfds.load('new_dataset', split = tfds.Split.TEST)    
    print("test set")
    for data in test_dataset:
        emb, label = data['embedding'], data['label']        
        one_hot_label = np.zeros(4)
        one_hot_label[label] = 1        
        x["test"].append(emb.numpy())
        y["test"].append(one_hot_label)
    
    recModel.set_trainset((np.array(x["train"]), np.array(y["train"])))
    recModel.set_testset((np.array(x["test"]), np.array(y["test"])))    
    
    recModel.fit()
    test_loss, test_acc = recModel.evaluate()    
    for i, val in enumerate(x["test"]):                
        print(recModel.predict(val))
        print(y["test"][i])
    recModel.model.save('recModel')
    print(f"evaluate => test_loss : {test_loss} , test_acc : {test_acc}", )                                 

if __name__ == '__main__':
    main()
