import numpy as np
import tensorflow as tf

def extract_features(img, target_size=224):
    model_path = "EdegInference/models"
    model = tf.saved_model.load(model_path)    
    img_resize = img.resize((target_size, target_size))                    
    x = np.array(img_resize, dtype = np.float32)        
    x = np.expand_dims(x, axis=0)     
    
    features = model(x)
    return features[0]

