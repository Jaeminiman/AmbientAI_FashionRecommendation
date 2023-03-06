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
import os
import cv2
import collections
from PIL import Image
import numpy as np
import recommender_training.vectorization as vec
from recommender_training.recommendation import *
import tensorflow as tf


from utils4smalls import tiles_location_gen, non_max_suppression, draw_object, reposition_bounding_box, set_resized_input

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])
def main():
    default_model_dir = '/all_models'
    object_detection_model = 'fashion_yolov5.pt'
    recommender_model = 'recModel'
    default_labels = 'clothes_label.json'
    parser = argparse.ArgumentParser()    
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))        
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default =7)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))    
    
    with open(args.labels, 'r') as f:
        label2id = json.load(f)
        
    label_list = list(label2id.keys())    

    
    cap = cv2.VideoCapture(args.input)    
    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #output video format arguments
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length
    yolov5_img_size = (640, 640)
    
    # Model load
    rec_saved = tf.saved_model.load(recommender_model)        
    recModel = RecommenderModelController(rec_saved)
    objectModel = tf.saved_model(object_detection_model)
    
    while cap.isOpened() and frames>0:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame        
        cv2.imshow('frame', cv2_im) # it's for realtime streaming
        if cv2.waitKey(1) & 0xFF == ord(' '):                    
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb_resize = cv2.resize(cv2_im_rgb, yolov5_img_size)            
            objs = get_objects(interpreter, args.threshold)[:args.top_k]        
            cv2_im = append_objs_to_img(cv2_im, yolov5_img_size, objs,label_list)             
            
            cropped = save_cropped_images(cv2_im_rgb, yolov5_img_size, objs, label_list)
            img, label = vectorization(cropped)
            if(label != None):
                if(label == "top"):
                    rec_score, rec_idx = recModel.botRecommend(img)                        
                else:
                    rec_score, rec_idx = recModel.topRecommend(img)

                rec_im = Image.open(recModel.closet2img[rec_idx])
                rec_im.show()
                rec_im.save('../recommeded_im/rec.jpg')
                                    
        out.write(frame)
        cv2.waitKey(1000)
        frames -=1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break
    
    
    
    cap.release()
    cv2.destroyAllWindows()   
        
def vectorization(cropped):
    top = None
    bot = None
    for img, obj in cropped:                
        label = obj.id
        top_label = [0,1,2,3,4,5]
        bottom_label = [6,7,8]
        exception_label = [9,10,11,12]
        if(label in top_label and top is None):
            top = vec.extract_features(img)
            
        elif(label in bottom_label and bot is None):
            bot = vec.extract_features(img)
    
        if((top is not None) and (bot is not None)):
            print("Take off one part of clothes for successful recommendation")
            return (None, None)
       
    
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
        print(f"saving crop image label : {label}, ranking : {rank}")
        cropped_image = im_rgb.crop((x0, y0, x1, y1))
        cropped_image.save(f'../crop_images/{label}.jpg')
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
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im



if __name__ == '__main__':
    main()
