from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import cv2
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from PIL import Image
from helpers import *
from pdb import set_trace as bp


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
