
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import align.detect_face
import glob

from pdb import set_trace as bp

from six.moves import xrange
from dataset.dataset_helpers import *

import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from models.resnet import *
from models.irse import *

from helpers import *

"""
#################################################################################
#################################################################################
#################################################################################
ARCFACE LOSS MS1-Celeb
#################################################################################

python3 app/export_embeddings_npy.py ./pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth ./data/golovan_112/ \
--mean_per_class 1 \
--is_aligned 1 \
--with_demo_images 1 \
--image_size 112 \
--image_batch 5 \
--embeddings_name embeddings_arcface_1.npy \
--labels_strings_array labels_strings_arcface_1.npy

"""

class FacesDataset(data.Dataset):
    def __init__(self, image_list, label_list, names_list, num_classes, is_aligned, image_size, margin, gpu_memory_fraction, demo_images_path=None):
        self.image_list = image_list
        self.label_list = label_list
        self.names_list = names_list
        self.num_classes = num_classes

        self.is_aligned = is_aligned
        self.demo_images_path = demo_images_path

        self.image_size = image_size
        self.margin = margin
        self.gpu_memory_fraction = gpu_memory_fraction

        self.static = 0

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')

        if self.is_aligned==1:
            image_data_rgb = np.asarray(data) # (160, 160, 3)
        else:
            image_data_rgb = load_and_align_data(img_path, self.image_size, self.margin, self.gpu_memory_fraction)


        ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=True)
        # bp()
        # print("\n\n")
        # print("### image_data_rgb shape: " + str(image_data_rgb.shape))
        # print("### CCROPPED shape: " + str(ccropped.shape))
        # print("### FLIPPED shape: " + str(flipped.shape))
        # print("\n\n")
        
        if self.demo_images_path is not None:
            ################################################
            ### SAVE Demo Images
            prefix = str(self.static)+ '_' + str(self.names_list[index]) 

            ## Save Matplotlib
            im_da = np.asarray(image_data_rgb)
            plt.imsave(self.demo_images_path + prefix + '.jpg', im_da)

            ## Save OpenCV
            # image_BGR = cv2.cvtColor(image_data_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(self.demo_images_path + prefix + '.png', image_BGR)

            self.static += 1
            ################################################

        
        # data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]
        return ccropped, flipped, label, name

    def __len__(self):
        return len(self.image_list)

def main(ARGS):
    
    np.set_printoptions(threshold=sys.maxsize)

    out_dir = ARGS.output_dir
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    images_dir=None
    if ARGS.with_demo_images==1:
        images_dir = os.path.join(os.path.expanduser(out_dir), 'demo_images/')
        if not os.path.isdir(images_dir):  # Create the out directory if it doesn't exist
            os.makedirs(images_dir)

    train_set = get_dataset(ARGS.data_dir)
    image_list, label_list, names_list = get_image_paths_and_labels(train_set)
    faces_dataset = FacesDataset(image_list=image_list, 
                                    label_list=label_list, 
                                    names_list=names_list, 
                                    num_classes=len(train_set), 
                                    is_aligned=ARGS.is_aligned, 
                                    image_size=ARGS.image_size, 
                                    margin=ARGS.margin, 
                                    gpu_memory_fraction=ARGS.gpu_memory_fraction,
                                    demo_images_path=images_dir)
    loader = torch.utils.data.DataLoader(faces_dataset, batch_size=ARGS.image_batch,
                                                shuffle=False, num_workers=ARGS.num_workers)


    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(ARGS.data_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if \
       os.path.isdir(os.path.join(path_exp, name))]


    ####### Model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IR_50([112, 112])
    model.load_state_dict(torch.load(ARGS.model, map_location='cpu'))
    model.to(device)
    model.eval()

    embedding_size = 512
    # emb_array = np.zeros((nrof_images, embedding_size))
    start_time = time.time()

    # ###### IMAGE
    # img_path = './data/test_image.png'
    # img = Image.open(img_path)
    # image_data = img.convert('RGB')
    # image_data_rgb = np.asarray(image_data) # shape=(160, 160, 3)  color_array=(255, 255, 255)
    # ccropped_im, flipped_im = crop_and_flip(image_data_rgb, for_dataloader=False)
    # feats_im = extract_norm_features(ccropped_im, flipped_im, model, device, tta = True)

    
########################################
    # nrof_images = len(loader.dataset)
    nrof_images = len(image_list)

    emb_array = np.zeros((nrof_images, embedding_size))
    # lab_array = np.zeros((nrof_images,))
    lab_array = np.zeros((0,0))

    # nam_array = np.chararray((nrof_images,))
    batch_ind = 0
    with torch.no_grad():
        for i, (ccropped, flipped, label, name) in enumerate(loader):

            ccropped, flipped, label = ccropped.to(device), flipped.to(device), label.to(device)

            # feats = model(data)
            feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
            
            # for j in range(len(ccropped)):
            #     # bp()
            #     dist = distance(feats_im.cpu().numpy(), feats[j].view(1,-1).cpu().numpy())
            #     # dist = distance(feats_im, feats[j])
            #     print("11111 Distance Eugene with {}  is  {}:".format(name[j], dist))

            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            # nam_array[lab] = name
            # lab_array[lab] = lab

            for j in range(len(ccropped)):
                emb_array[j+batch_ind, :] = emb[j, :]
            
            lab_array = np.append(lab_array,lab)
            
            # print("\n")
            # for j in range(len(ccropped)):
            #     dist = distance(feats_im.cpu().numpy(), np.expand_dims(emb_array[j+batch_ind], axis=0))
            #     # dist = distance(feats_im, feats[j])
            #     print("22222 Distance Eugene with {}  is  {}:".format(name[j], dist))
            # print("\n")


            batch_ind += len(ccropped)

            percent = round(100. * i / len(loader))
            print('.completed {}%  Run time: {}'.format(percent, timedelta(seconds=int(time.time() - start_time))), end='\r')

        print('', end='\r')
    print(60*"=")
    print("Done with embeddings... Exporting")

    if ARGS.mean_per_class==1:
        print("Exporting embeddings mean for class")

        label_strings = np.array(label_strings)
        label_strings_all = label_strings[label_list]

        all_results_dict = {}
        for j in range(nrof_images):
            embedding = emb_array[j,:]
            label = label_strings_all[j]
            if label in all_results_dict: # if label value in dictionary
                arr = all_results_dict.get(label)
                arr.append(embedding)
            else:
                all_results_dict[label] = [embedding]

        ## Saving mean
        nrof_classes = len(classes)
        emb_array_out = np.zeros((nrof_classes, embedding_size))
        lab_array_out = np.zeros((0,0))
        label_strings_out = []
        
        embedding_index = 0
        for key, embeddings_arr in all_results_dict.items():

            numpy_arr = np.array(embeddings_arr)
            mean = np.mean(numpy_arr, axis=0)
            emb_array_out[embedding_index] = mean

            lab_array_out = np.append(lab_array_out, embedding_index)
            embedding_index += 1

            label_strings_out.append(key)

        #   export emedings and labels
        np.save(out_dir + ARGS.embeddings_name, emb_array_out)
        # np.save(out_dir + ARGS.labels, lab_array_out)

        label_strings = np.array(label_strings_out)
        np.save(out_dir + ARGS.labels_strings_array, label_strings)

    else:
        print("Exporting All embeddings")
        #   export emedings and labels
        np.save(out_dir + ARGS.embeddings_name, emb_array)
        # np.save(out_dir + ARGS.labels, lab_array)

        label_strings = np.array(label_strings)
        np.save(out_dir + ARGS.labels_strings_array, label_strings[label_list])

    total_time = timedelta(seconds=int(time.time() - start_time))
    print(60*"=")
    print('All done. Total time: ' + str(total_time))


def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    print('🎃  Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    print(image_path)
    img = misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

    img = aligned
    
    return img

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='pth model file')
    parser.add_argument('data_dir', type=str, help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--output_dir', type=str, help='Dir where to save all embeddings and demo images', default='output_arrays/')
    parser.add_argument('--mean_per_class', type=int, help='Export mean of all embeddings for each class 0:False 1:True', default=1)
    parser.add_argument('--is_aligned', type=int, help='Is the data directory already aligned and cropped? 0:False 1:True', default=1)
    parser.add_argument('--with_demo_images', type=int, help='Embedding Images 0:False 1:True', default=1)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 64.', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str, help='Enter string of which the embeddings numpy array is saved as.', default='embeddings.npy')
    parser.add_argument('--labels_strings_array', type=str, help='Enter string of which the labels as strings numpy array is saved as.', default='label_strings.npy')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
