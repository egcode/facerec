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
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def add_overlays(frame, faces, ARGS):
    font_scale = ARGS.font_size
    # font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    rectangle_bgr = (0, 0, 0)
    face_rectangle_thick = 2
    bg_margin = 5
    label_y_offset = bg_margin

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


            final_name = name
            if ARGS.show_distance==1:
                final_name = name + " " + str(round(face.distance, 2))
            
            # ######## Centered
            # # text bg
            # (text_width, text_height) = cv2.getTextSize(final_name, font, fontScale=font_scale, thickness=1)[0]
            # box_coords = ((face_bb[0]-bg_margin, face_bb[3]+(text_height+bg_margin+label_y_offset)), (face_bb[0] + text_width+bg_margin, face_bb[3]-(bg_margin-label_y_offset)))
            # cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            
            # # text
            # cv2.putText(frame, final_name, (face_bb[0], face_bb[3]+(text_height+label_y_offset)),
            #         font, font_scale, color,
            #         thickness=2, lineType=2)

            ####### Aligned to Right 
            # text bg
            (text_width, text_height) = cv2.getTextSize(final_name, font, fontScale=font_scale, thickness=1)[0]
            box_coords = ((face_bb[0]-(bg_margin-bg_margin), face_bb[3]+(text_height+bg_margin+label_y_offset)), (face_bb[0] + text_width+(bg_margin+bg_margin), face_bb[3]-(bg_margin-label_y_offset)))
            cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            
            # text
            cv2.putText(frame, final_name, (face_bb[0]+(bg_margin), face_bb[3]+(text_height+label_y_offset)),
                    font, font_scale, color,
                    thickness=2, lineType=2)

            # Main face rectangle on top
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          color, face_rectangle_thick)

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
