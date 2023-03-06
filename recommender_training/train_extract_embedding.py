import random
from PIL import Image
import glob
import os
import pandas as pd
import json
import vectorization as vec
import numpy as np

def extract_embedding():
    default_model_dir = 'all_models'    
    default_labels = 'clothes_label.json'
    labels = os.path.join(default_model_dir, default_labels)


    print('Loading with {} labels.'.format(labels))    
    with open(labels, 'r') as f:
        label2id = json.load(f)
    label_list = list(label2id.keys())
    # input : cropped_img
    # output : embeddings with depth 100 
    base_path = 'clothes-detection-yolov5/runs/detect' 
    outfit_lables = ["military", "minimal", "formal","street"]

        
    input = []    
    for i, outfit_label in enumerate(outfit_lables):                           
            print(f"진행 상황을 알려드릴게요 : {outfit_label}")
            print("좀만 더 화이팅 하자구용~")
            path = os.path.join(base_path,outfit_label)        
            outfits = glob.glob(path + "/*")        
            for outfit in outfits:
                
                imgs = glob.glob(outfit+ "/*.jpg")
                if(len(imgs) == 2):
                    cropped = []
                    for img in imgs:
                        im = Image.open(img)
                        label = img.split("/")[-1][:-4]                    
                        if(label in label_list):
                            label_idx = label2id[label]
                            cropped.append((im, label_idx))                
                    vec = vectorization(cropped)
                    print("추출 완료 !")
                    
                    if(vec is not None):                    
                        top, bot = vec           
                        con = np.concatenate((top,bot), axis = 0)                
                        con = ':'.join(str(e) for e in con)                    
                        input.append((i, con))                    
                else:
                    if os.path.isdir(outfit + "/"):
                        for img in imgs:                        
                            os.remove(img)
                        os.rmdir(outfit + "/")
                        print("제거 완료")

    
    # input save in csv file
    random.shuffle(input)
    seventy_percentage = int(len(input) * 0.7)
    train_data = input[:seventy_percentage]
    test_data = input[seventy_percentage:]    
    train_df = pd.DataFrame(train_data, columns = ['label', 'embedding'])    
    train_df.to_csv('all_models/data/train/embedding.csv', index=False)
    test_df = pd.DataFrame(test_data, columns = ['label', 'embedding'])    
    test_df.to_csv('all_models/data/test/embedding.csv', index=False)


def vectorization(cropped):
    top = None
    bot = None
    for img, id in cropped:
        # Image to np.ndarray        
                
        top_label = [0,1,2,3,4,5]
        bottom_label = [6,7,8]
        exception_label = [9,10,11,12]
        if(id in top_label and top is None):
            top = vec.extract_features(img)
            
        elif(id in bottom_label and bot is None):
            bot = vec.extract_features(img)
    
    if((top is not None) and (bot is not None)):            
        return (top, bot)
    else:
        return None