from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS MS1-Celeb
#################################################################################

# Eugene Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/eugene1.png \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1

# Liuba Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/liuba1.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1

# Julia Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/julia1.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1


# Curen Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/curen1.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1

# Jeffrey Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/jeffrey2.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1


# David Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/david1.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1

# Alex Image
python3 app/compare_embeddings_with_image.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_path ./data/test_images/alex3.jpg \
--h5_name ./out_embeddings/golovan_112.h5 \
--distance_metric 1

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
from scipy import spatial

from models.resnet import *
from models.irse import *
from helpers import *
from pdb import set_trace as bp
import h5py                                                                                                                                                                                   

def main(ARGS):

    ###### IMAGE
    img_path = ARGS.image_path
    img = Image.open(img_path)
    image_data = img.convert('RGB')
    image_data_rgb = np.asarray(image_data) # shape=(112, 112, 3)  color_array=(255, 255, 255)
    ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=False)
    # image_data.save('pilllllllll.png')
    
    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()
    #########################################

    with torch.no_grad():
        feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
        feats = feats.cpu().numpy()


    #########################################

    all_results_dict = {}

    with h5py.File(ARGS.h5_name, 'r') as f:
        for person in f.keys():
            embedding_premade = f[person]['embedding'][:]

            label = person

            dist = distance(feats, embedding_premade, ARGS.distance_metric)
            # dist = spatial.distance.cosine(feats, embedding_premade)

            label = person

            print("Distance with {}: {}".format(label, dist))

            if label in all_results_dict: # if label value in dictionary
                arr = all_results_dict.get(label)
                arr.append(dist)
            else:
                all_results_dict[label] = [dist]
            # print("candidate: " + str(i) + " distance: " + str(dist) + " with " + label_strings[j])


    print("======EMBEDDINGS ALL RESULTS===========")
    for key, distances_arr in all_results_dict.items():
        print("Average Distance for {} : {}".format(key, np.mean(distances_arr)))
    print("=================")
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pth model file')
    parser.add_argument('--image_path', type=str, help='image to compare')
    parser.add_argument('--h5_name', type=str, help='h5 file name', default='./out_embeddings/golovan_112.h5')
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

