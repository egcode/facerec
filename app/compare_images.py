from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''

#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS - MS-Celeb
#################################################################################

##### NOT SAME        without extract_feature distance = [0.9698452893644571] 
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Liuba/l2.jpg \
--image_two_path ./data/golovan_112/Julia/j6.jpg \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.9670979678630829]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Alex/a8.jpg \
--image_two_path ./data/golovan_112/Julia/j2.jpg \
--distance_metric 1

##### NOT SAME        without extract_feature distance = [0.997516609262675]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Eugene/e5.jpg \
--image_two_path ./data/golovan_112/Julia/j12.jpg \
--distance_metric 1


##### NOT SAME        without extract_feature distance = [0.8797276243567467]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Eugene/e14.jpg \
--image_two_path ./data/golovan_112/Alex/a3.jpg \
--distance_metric 1






##### SAME            without extract_feature distance = [0.4535517692565918]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Julia/j5.jpg \
--image_two_path ./data/golovan_112/Julia/j14.jpg \
--distance_metric 1

##### SAME            without extract_feature distance = [0.659491240978241]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Julia/j2.jpg \
--image_two_path ./data/golovan_112/Julia/j3.jpg \
--distance_metric 1

##### SAME            without extract_feature distance = [0.3201570510864258]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Eugene/e8.jpg \
--image_two_path ./data/golovan_112/Eugene/e1.jpg \
--distance_metric 1

##### SAME            without extract_feature distance = [0.42241138219833374]
python3 app/compare_images.py \
--model ./pth/IR_50_MODEL_arcface_ms1celeb_epoch88_lfw9957.pth \
--image_one_path ./data/golovan_112/Alex/a9.jpg \
--image_two_path ./data/golovan_112/Alex/a2.jpg \
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

from models.resnet import *
from models.irse import *
from helpers import *
from pdb import set_trace as bp

def main(ARGS):

    ###### IMAGE 1111
    img_path1 = ARGS.image_one_path
    img1 = Image.open(img_path1)
    image_data1 = img1.convert('RGB')
    image_data_rgb_1 = np.asarray(image_data1) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    ccropped_1, flipped_1 = crop_and_flip(image_data_rgb_1, for_dataloader=False)
    # image_data1.save('pilllllllll.png')

    ###### IMAGE 2222
    img_path2 = ARGS.image_two_path
    img2 = Image.open(img_path2)
    image_data2 = img2.convert('RGB')
    image_data_rgb_2 = np.asarray(image_data2) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    ccropped_2, flipped_2 = crop_and_flip(image_data_rgb_2, for_dataloader=False)

    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()
    #########################################

    with torch.no_grad():
        feats_1 = extract_norm_features(ccropped_1, flipped_1, model, device, tta = True)
        feats_1 = feats_1.cpu().numpy()

        feats_2 = extract_norm_features(ccropped_2, flipped_2, model, device, tta = True)
        feats_2 = feats_2.cpu().numpy()

    dist = distance(feats_1, feats_2, ARGS.distance_metric)

    print("======DISTANCE===========")
    print(dist)
    print("=================")
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='pth model file')
    parser.add_argument('--image_one_path', type=str, help='ONE')
    parser.add_argument('--image_two_path', type=str, help='TWO')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

