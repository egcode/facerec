
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
from helpers import *
import h5py
from datetime import datetime, timedelta

### Requirement: name, path, embedding

'''
person1_name
    label    0
    person1_subgroup_1
        file_path    '/path/to/file1'
        embedding    [4.5, 2.1, 9.9]
    person1_subgroup_2
        file_path    '/path/to/file123'
        embedding    [84.5, 32.32, 10.1]

person2_name
    label    1
    person2_subgroup_1
        file_path    '/path/to/file4444'
        embedding    [1.1, 2.1, 2.9]
    person2_subgroup_2
        file_path    '/path/to/file1123123'
        embedding    [3.0, 41.1, 56.621]

'''


"""

#################################################################################
#################################################################################
#################################################################################
COSFACE LOSS-Eugene Casia
#################################################################################

python3 dataset_cleanup/export_dataset_embeddings.py \
./pth/IR_50_MODEL_arcface_ms1celeb_epoch90_lfw9962.pth \
./data/embedding_test/ \
--model_type IR_50 \
--image_batch 5 \
--h5_name dataset.h5

"""

class FacesDataset(data.Dataset):
    def __init__(self, image_list, label_list, names_list, num_classes):
        self.image_list = image_list
        self.label_list = label_list
        self.names_list = names_list
        self.num_classes = num_classes

        self.static = 0

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = Image.open(img_path)
        data = img.convert('RGB')

        image_data_rgb = np.asarray(data) # (112, 112, 3)

        ccropped, flipped = crop_and_flip(image_data_rgb, for_dataloader=True)        
        # data = self.transforms(data)
        label = self.label_list[index]
        name = self.names_list[index]

        apsolute_path = os.path.abspath(img_path)
        return ccropped, flipped, label, name, apsolute_path

    def __len__(self):
        return len(self.image_list)

def writePerson(h5_filename, person_name, person_label, image_name, image_path, embedding):
    '''
    Example;
        h5_filename='output_dataset/dataset.h5'
        person_name='Alex'
        person_label=0
        image_name='a1.jpg'
        image_path=/path/to/Alex/a1.jpg'
        embedding=[-1.40146054e-02,  2.31648367e-02, -8.39150697e-02......]
    '''
    with h5py.File(h5_filename, 'a') as f:
        
        if person_name in f.keys():
            person_subgroup = f[person_name].create_group(image_name)
            person_subgroup.create_dataset('embedding',  data=embedding)  
            person_subgroup.attrs["file_path"] = np.string_(image_path)

        else:
            person_grp = f.create_group(person_name)
            person_grp.attrs['label'] = person_label

            person_subgroup = person_grp.create_group(image_name)
            person_subgroup.create_dataset('embedding',  data=embedding)  
            person_subgroup.attrs["file_path"] = np.string_(image_path)


def main(ARGS):
    
    np.set_printoptions(threshold=sys.maxsize)

    out_dir = ARGS.output_dir
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    if os.path.isfile(out_dir+ARGS.h5_name): # remove previous file if exists
        os.remove(out_dir+ARGS.h5_name)

    train_set = get_dataset(ARGS.data_dir)
    image_list, label_list, names_list = get_image_paths_and_labels(train_set)
    faces_dataset = FacesDataset(image_list=image_list, 
                                    label_list=label_list, 
                                    names_list=names_list, 
                                    num_classes=len(train_set))
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
    # emb_array = np.zeros((nrof_images, embedding_size))
    start_time = time.time()
    
########################################
    nrof_images = len(image_list)

    with torch.no_grad():
        for i, (ccropped, flipped, label, name, absolute_paths) in enumerate(loader):

            ccropped, flipped, label = ccropped.to(device), flipped.to(device), label.to(device)

            feats = extract_norm_features(ccropped, flipped, model, device, tta = True)
            
            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            for j in range(len(ccropped)):

                #params
                person_embedding = emb[j, :]
                person_label = lab[j]
                person_name = name[j]
                image_path = absolute_paths[j]
                image_name = os.path.basename(absolute_paths[j])
                writePerson(out_dir+ARGS.h5_name, person_name, person_label, image_name, image_path, person_embedding)
            
            percent = round(100. * i / len(loader))
            print('.completed {}%  Run time: {}'.format(percent, timedelta(seconds=int(time.time() - start_time))), end='\r')

        print('', end='\r')
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(60*"=")
    print('All done. Total time: ' + str(total_time))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='pth model file')
    parser.add_argument('data_dir', type=str, help='Directory containing images.')
    parser.add_argument('--output_dir', type=str, help='Dir where to save dataset', default='data/')
    parser.add_argument('--image_batch', type=int, help='Number of images stored in memory at a time. Default 64.', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    parser.add_argument('--h5_name', type=str, help='h5 file name', default='dataset.h5')
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
