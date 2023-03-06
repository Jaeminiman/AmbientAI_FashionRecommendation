import numpy as np
#import tensorflow as tf
from pycoral.adapters.common import input_size
from pycoral.adapters.common import output_tensor
from new.pycoral.pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def extract_features(img, target_size=224):
    # Load the TFLite model and allocate tensors.
    interpreter = make_interpreter("/home/mendel/examples-camera/all_models/quantized_featureExt.tflite")
    interpreter.allocate_tensors()
    img_resized = img.resize((target_size, target_size))
    img_array = np.array(img_resized, dtype=np.uint8)
    run_inference(interpreter, img_array.tobytes())
    output = output_tensor(interpreter, 0)[0]
    
    return output
