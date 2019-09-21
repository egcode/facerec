
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS - MS1-Celeb
#################################################################################

python3 app/live_cam_face_recognition.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth \
--unknown_face unknown \
--max_threshold 0.6 \
--distance_metric 1 \
--h5_name ./out_embeddings/golovan_112.h5
'''

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
import cv2
from imutils.video import VideoStream
import imutils
import time
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image
from models.resnet import *
from models.irse import *
from helpers import *
from pdb import set_trace as bp
import h5py                                                                                                                                                                                   

class Face:
    def __init__(self):
        self.name = None
        self.distance = None
        self.bounding_box = None
        self.image = None
        self.embedding = None
        self.all_results_dict = {}

    def parse_all_results_dict(self, max_threshold):
        average_dist_dict = {}
        for key, distances_arr in self.all_results_dict.items():
            average_dist_dict[key] = np.mean(distances_arr)

        name = min(average_dist_dict, key=average_dist_dict.get) #get minimal value from dictionary
        self.distance = average_dist_dict[name]

        if average_dist_dict[name] < max_threshold: 
            self.name = name

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32, gpu_memory_fraction = 0.3):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image, image_size):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            
            face_image = aligned[:,:,::-1] ## BRG -> RGB

            face = Face()
            face.image = face_image
            face.bounding_box = bb
            faces.append(face)

        return faces

def main(ARGS):
  
    vs = VideoStream(src=0).start() # regular webcam camera
    # vs = VideoStream(usePiCamera=True).start() # raspberry pi camera 

    detect = Detection()
      
    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
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

def add_overlays(frame, faces, ARGS):
    color_positive = (0, 255, 0)
    color_negative = (0, 0, 255)
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)

            color = color_negative
            name = ARGS.unknown_face
            if face.name is not None and face.distance is not None:
                color = color_positive
                name = face.name

            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          color, 2)

            final_name = name
            if ARGS.show_distance==1:
                final_name = name + " " + str(round(face.distance, 2))
            
            cv2.putText(frame, final_name, (face_bb[0], face_bb[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=2)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pth model file')
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
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

