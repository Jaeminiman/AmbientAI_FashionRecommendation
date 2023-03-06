# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import collections
from PIL import Image
import numpy as np
import vectorization as vec
from recommendation import *

import json

from pycoral.adapters.common import input_size
from new.pycoral.pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from utils4smalls import tiles_location_gen, non_max_suppression, draw_object, reposition_bounding_box, set_resized_input

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])
def main():
    default_model_dir = '../all_models'
    default_model = 'efficientdet-lite-fashion_edgetpu.tflite'
    default_labels = 'clothes_label1.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default =10)
    parser.add_argument('--wtr', help='what to recommend', type=int, default = 0)
    parser.add_argument('--style', help='which style', type=int, default = 0)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    with open(args.labels, 'r') as f:
        label2id = json.load(f)
    label_list = list(label2id.keys())
    wtr = args.wtr
    style = args.style
    inference_size = input_size(interpreter)

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #output video format arguments
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length
    
    # Recommender Model load
    recModel = RecommenderModel('/home/mendel/examples-camera/all_models/quantized_recModel_edgetpu.tflite')
    
    while cap.isOpened() and frames>0:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame        
        cv2.imshow('frame', cv2_im) # it's for realtime streaming
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): 
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb2 = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb2.tobytes())
            objs = get_objects(interpreter, args.threshold)[:args.top_k]        
            cv2_im = append_objs_to_img(cv2_im, inference_size, objs,label_list)             
            
            cropped = save_cropped_images(cv2_im_rgb, inference_size, objs, label_list)
            img, label = vectorization(cropped, wtr)

            if(label != None):
                if(label == "top"):
                    rec_score, rec_idx = recModel.botRecommend(img, style)
                else:
                    rec_score, rec_idx = recModel.topRecommend(img, style)

                rec_im = Image.open(recModel.closet2img[int(rec_idx)])
                open_cv_image = np.array(rec_im) 
                open_cv_image1 = open_cv_image[:, :, ::-1].copy() 
                #cv2.imshow('frame', open_cv_image1)
                rec_scale = min(frame_width/np.shape(open_cv_image)[1], frame_height/np.shape(open_cv_image)[0])
                rec = rec_im.resize((int(rec_scale*np.shape(open_cv_image)[1]), int(rec_scale*np.shape(open_cv_image)[0])))
                rec = np.array(rec)                
                rc = np.pad(rec, ((int((frame_height - np.shape(rec)[0])/2),frame_height - np.shape(rec)[0]-int((frame_height - np.shape(rec)[0])/2)), (int((frame_width - np.shape(rec)[1])/2),frame_width - np.shape(rec)[1]-int((frame_width - np.shape(rec)[1])/2)), (0,0)), 'constant')                                                
                zero = cv2.cvtColor(rc, cv2.COLOR_BGR2RGB)
                zero1 = zero[:, :, ::-1].copy()
                cv2.imshow('frame', zero)
                while frames>0:
                    out.write(zero)
                    frames -= 1
                cv2.waitKey(2000)
                rec_im.save('/home/mendel/examples-camera/opencv/recommended_im/rec.jpg')
                break

                

        #out.write(frame)
        #cv2.waitKey(2000)
        frames -= 1

#        if cv2.waitKey(1) & 0xFF == ord('q'):   
#            print("pressed q")
#            break

    cap.release()
    cv2.destroyAllWindows()   
        
def vectorization(cropped, wtr):
    top = None
    bot = None
    for img, obj in cropped:                        
        label = obj.id        
        top_label = [0,1,2,3,4,5]
        bottom_label = [6,7,8]
        exception_label = [9,10,11,12]
        
        if(label in top_label and wtr == 1):
            top = vec.extract_features(img)
            
        elif(label in bottom_label and wtr == 0):
            bot = vec.extract_features(img)
        

        '''
        if(label in top_label and top is None):
            top = vec.extract_features(img)
            
        elif(label in bottom_label and bot is None):
            bot = vec.extract_features(img)
    
        if((top is not None) and (bot is not None)):
            print("Take off one part of clothes for successful recommendation")
            return (None, None)
        '''
    
    if((top is None) and (bot is not None)):
        return (bot, "bot")
    elif((top is not None) and (bot is None)):
        return (top, "top")
    else:
        print("Take on at least one part of clothes for successful recommendation")    
        return (None, None)
    
        
        
        
        
    
def save_cropped_images(cv2_im_rgb, inference_size, objs, labels):
    height, width, channels = cv2_im_rgb.shape
    im_rgb = Image.fromarray(cv2_im_rgb.astype('uint8'), 'RGB')
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    
    output = []
    for rank, obj in enumerate(objs):
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        label = labels[obj.id]
        print(obj.id)
        print(label)
        print(f"saving crop image label : {label}, ranking : {rank}")
        cropped_image = im_rgb.crop((x0, y0, x1, y1))
        cropped_image.save(f'/home/mendel/examples-camera/opencv/crop_images/{label}.jpg')
        output.append((cropped_image, obj))
    return output
        
        

        
def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels[obj.id])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im



if __name__ == '__main__':
    main()

