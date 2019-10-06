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
from helpers import *


"""
python3 app/export_embeddings.py \
--model_path ./data/pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth \
--data_dir ./data/dataset_got/dataset_lanister_raw/ \
--output_dir data/out_embeddings/  \
--model_type IR_50 \
--is_aligned 0 \
--with_demo_images 1 \
--image_size 112 \
--image_batch 5 \
--h5_name dataset_lanister.h5
"""

def writePersonMeanEmbeddingFile(h5_filename, person_name, mean_embedding):
    '''
    =====================================
    *** Mean embedding h5 file structure:
    person1_name
        embedding    [4.5, 2.1, 9.9]
    person2_name
        embedding    [3.0, 41.1, 56.621]
    =====================================

    Parameters;
        h5_filename='data/dataset.h5'
        person_name='Alex'
        mean_embedding=[-1.40146054e-02,  2.31648367e-02, -8.39150697e-02......]
    '''
    with h5py.File(h5_filename, 'a') as f:
        person_grp = f.create_group(person_name)
        person_grp.create_dataset('embedding',  data=mean_embedding)  


def writePersonTempFile(temp_h5_filename, person_name, image_temp_name, embedding):
    '''
    =====================================
    *** temp h5 file structure:
    person1_name
        person1_subgroup_imagetempname_1
            embedding    [4.5, 2.1, 9.9]
        person1_subgroup_imagetempname_2
            embedding    [84.5, 32.32, 10.1]

    person2_name
        person2_subgroup_imagetempname_1
            embedding    [1.1, 2.1, 2.9]
        person2_subgroup_imagetempname_2
            embedding    [3.0, 41.1, 56.621]
    =====================================

    Parameters;
        temp_h5_filename='data/temp_dataset.h5'
        person_name='Alex'
        image_temp_name='a1.jpg'
        embedding=[-1.40146054e-02,  2.31648367e-02, -8.39150697e-02......]
    '''
    with h5py.File(temp_h5_filename, 'a') as f:
        
        if person_name in f.keys():
            person_subgroup = f[person_name].create_group(image_temp_name)
            person_subgroup.create_dataset('embedding',  data=embedding)  

        else:
            person_grp = f.create_group(person_name)

            person_subgroup = person_grp.create_group(image_temp_name)
            person_subgroup.create_dataset('embedding',  data=embedding)  

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

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')

        if self.is_aligned==1:
            image_data_rgb = np.asarray(data) # (112, 112, 3)
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
            image_name = str(self.names_list[index]) + '_' + str(os.path.basename(img_path))
            ## Save Matplotlib
            im_da = np.asarray(image_data_rgb)
            plt.imsave(self.demo_images_path + image_name, im_da)

            ## Save OpenCV
            # image_BGR = cv2.cvtColor(image_data_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(self.demo_images_path + prefix + '.png', image_BGR)

            ################################################

        
        # data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]
        apsolute_path = os.path.abspath(img_path)

        return ccropped, flipped, label, name, apsolute_path

    def __len__(self):
        return len(self.image_list)

def main(ARGS):
    
    # np.set_printoptions(threshold=sys.maxsize)

    out_dir = ARGS.output_dir
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)
    else:
        if os.path.exists(os.path.join(os.path.expanduser(out_dir), ARGS.h5_name)):
            os.remove(os.path.join(os.path.expanduser(out_dir), ARGS.h5_name))
        

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


    embedding_size = 512
    start_time = time.time()
    
########################################
    temp_file = out_dir+"temp_"+ARGS.h5_name
    with torch.no_grad():
        for i, (ccropped, flipped, label, name, absolute_paths) in enumerate(loader):

            ccropped, flipped, label = ccropped.to(device), flipped.to(device), label.to(device)

            feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
            
            emb = feats.cpu().numpy()

            for j in range(len(ccropped)):

                #params
                person_embedding = emb[j, :]
                person_name = name[j]
                image_temp_name = os.path.basename(absolute_paths[j])
                writePersonTempFile(temp_file, person_name, image_temp_name, person_embedding)

            percent = round(100. * i / len(loader))
            print('.completed {}%  Run time: {}'.format(percent, timedelta(seconds=int(time.time() - start_time))), end='\r')

        print('', end='\r')
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(60*"=")
    print('Extracting embeddings done. time: ' + str(total_time))

###########################################################
### Extracting MEAN embedding for each person
    '''
    =====================================
    *** temp h5 file structure:
    person1_name
        person1_subgroup_imagetempname_1
            embedding    [4.5, 2.1, 9.9]
        person1_subgroup_imagetempname_2
            embedding    [84.5, 32.32, 10.1]

    person2_name
        person2_subgroup_imagetempname_1
            embedding    [1.1, 2.1, 2.9]
        person2_subgroup_imagetempname_2
            embedding    [3.0, 41.1, 56.621]
    =====================================
    '''

    if not os.path.isfile(temp_file):
        assert "temp h5 file is not exist"

    print('Extracting mean embeddings...\n')

    # Data for each person in temp file
    with h5py.File(temp_file, 'r') as f:
        for person in f.keys():
            # print("\npersonName: " + str(person))

            nrof_images = len(f[person].keys())
            embedding_size = 512
            embeddings_array = np.zeros((nrof_images, embedding_size))
            # label_strings_array = []

            print('For {} extracted {} embeddings'.format(person, nrof_images))
            # print("\tembedding array shape: " + str(embeddings_array.shape))
            # print("\tnumber of images: " + str(nrof_images) + "  embedding size: " + str(embedding_size))

            for i, subgroup in enumerate(f[person].keys()):
                # print("\tlabel: " + str(i))
                embeddings_array[i, :] = f[person][subgroup]['embedding'][:]
                # label_strings_array.append(str(subgroup))
                # print("\timage_name: " + str(subgroup))
                # print("\tembedding: " + str(f[person][subgroup]['embedding'][:]))

            mean_embedding = np.mean(embeddings_array, axis=0)
            writePersonMeanEmbeddingFile(out_dir+ARGS.h5_name, person, mean_embedding)

    print('\nExtracting mean embeddings done. time: ' + str(total_time))
    if os.path.exists(temp_file):
        os.remove(temp_file)
    else:
        print("Failed to remove temp h5 file {}".format(temp_file))
    
    print(60*"=")
    print('All done. time: ' + str(total_time))


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
    parser.add_argument('--model_path', type=str, help='pth model file')
    parser.add_argument('--data_dir', type=str, help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.')
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--output_dir', type=str, help='Dir where to save all embeddings and demo images', default='data/out_embeddings/')
    parser.add_argument('--is_aligned', type=int, help='Is the data directory already aligned and cropped? 0:False 1:True', default=1)
    parser.add_argument('--with_demo_images', type=int, help='Embedding Images 0:False 1:True', default=1)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--margin', type=int, help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 64.', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    #   numpy file Names
    parser.add_argument('--h5_name', type=str, help='h5 file name', default='dataset.h5')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
