
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from app.detection import Detection, Face, add_overlays
import cv2
from imutils.video import VideoStream
import imutils
import time
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image
from helpers import *
from pdb import set_trace as bp
import h5py                                                                                                                                                                                   

'''
python3 app/face_recognition_live_cam.py \
--model_path ./data/pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth \
--model_type IR_50 \
--unknown_face unknown \
--max_threshold 0.6 \
--distance_metric 1 \
--font_size 0.5 \
--h5_name ./data/out_embeddings/golovan_112.h5
'''

def main(ARGS):
  
    vs = VideoStream(src=0).start() # regular webcam camera
    # vs = VideoStream(usePiCamera=True).start() # raspberry pi camera 

    detect = Detection()
      
    ####### Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    print("Use CUDA: " + str(use_cuda))
    print('Model type: %s' % ARGS.model_type)
    model = get_model(ARGS.model_type, ARGS.input_size)
    if use_cuda:
        model.load_state_dict(torch.load(ARGS.model_path))
    else:
        model.load_state_dict(torch.load(ARGS.model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        #########################################

        faces = detect.find_faces(frame, ARGS.image_size)
        for face in faces:
            face.distance = 9
        for i, face in enumerate(faces):
            pil_image = Image.fromarray(face.image, mode='RGB')
    
            image_data_rgb = np.asarray(pil_image) # shape=(112, 112, 3)  color_array=(255, 255, 255)
            ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=False)

            with torch.no_grad():
                feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
                face.embedding = feats.cpu().numpy()

        # nrof_premade = embeddings_premade.shape[0]

        for i in range(len(faces)):
            with h5py.File(ARGS.h5_name, 'r') as f:
                for person in f.keys():
                    embedding_premade = f[person]['embedding'][:]

                    face = faces[i]
                    # dist = np.sqrt(np.sum(np.square(np.subtract(face.embedding, embeddings_premade[j,:]))))
                    dist = distance(face.embedding, embedding_premade.reshape((1, 512)), ARGS.distance_metric)
                    # print("Distance: {}".format(dist))

                    label = person
                    if label in face.all_results_dict: # if label value in dictionary
                        arr = face.all_results_dict.get(label)
                        arr.append(dist)
                    else:
                        face.all_results_dict[label] = [dist]
                    # print("candidate: " + str(i) + " distance: " + str(dist) + " with " + labels_strings_array[j])
        
        for i in range(len(faces)):
            # print("FACE :" + str(i))
            # print(faces[i].all_results_dict)
            faces[i].parse_all_results_dict(ARGS.max_threshold)

        add_overlays(frame, faces, ARGS)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

                    
                   
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='pth model file')
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--embeddings_premade', type=str, help='Premade embeddings array .npy format')
    parser.add_argument('--labels_strings_array', type=str, help='Premade label strings array .npy format')
    parser.add_argument('--show_distance', type=int, help='Show distance on label 0:False 1:True', default=0)
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--unknown_face', type=str, help='Unknown face will be labeled with this string', default='unknown')
    parser.add_argument('--max_threshold', type=float, help='If distance larger than this value, class labeled as unknown_face parameter', default=0.6)
    parser.add_argument('--h5_name', type=str, help='h5 file name', default='./output_arrays/dataset.h5')
    parser.add_argument('--font_size', type=float, help='Face label font size', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

