from vectorization import *
import glob
from PIL import Image
import json
import os
import numpy as np

closet = dict({"top" : [], "bot" : [] })
base_path = 'closet/img'         
clothes_types = glob.glob(base_path + "/*")
for clothes_path in clothes_types:        
        clothes_type = clothes_path.split("/")[-1] # top or bot
        imgs = glob.glob(clothes_path + "/*.jpg")
        output = []
        for img in imgs:
                im = Image.open(img)
                idx = img.split("/")[-1][:-4]
                print(f"image index : {idx}")
                embedding = list(np.array(extract_features(im)))
                embedding = [str(x) for x in embedding]               
                closet[clothes_type].append(dict({"idx" :idx , "embed" : embedding}))                

with open('closet/closet.json', 'w') as f:
        json.dump(closet, f, indent=2)