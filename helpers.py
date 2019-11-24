from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from six import iteritems
from subprocess import Popen, PIPE
from datetime import datetime
import cv2
from PIL import Image
import numpy as np
from scipy import spatial
import math
from models.resnet import *
from models.irse import *
from models.mobilenet import *
from models.lightnet import *

from pdb import set_trace as bp

###################################################################
## Model Helpers

def get_model(model_type, input_size):
    if model_type == 'LightNet':
        return LightNet(input_size)
    if model_type == 'MobileNet':
        return MobileNet(input_size)
    if model_type == 'ResNet_50':
        return ResNet_50(input_size)
    elif model_type == 'ResNet_101':
        return ResNet_101(input_size)
    elif model_type == 'ResNet_152':
        return ResNet_152(input_size)
    elif model_type == 'IR_50':
        return IR_50(input_size)
    elif model_type == 'IR_101':
        return IR_101(input_size)
    elif model_type == 'IR_152':
        return IR_152(input_size)
    elif model_type == 'IR_SE_50':
        return IR_SE_50(input_size)
    elif model_type == 'IR_SE_101':
        return IR_SE_101(input_size)
    elif model_type == 'IR_SE_152':
        return IR_SE_152(input_size)
    else:
        raise AssertionError('Unsuported model_type {}. We only support: [\'ResNet_50\', \'ResNet_101\', \'ResNet_152\', \'IR_50\', \'IR_101\', \'IR_152\', \'IR_SE_50\', \'IR_SE_101\', \'IR_SE_152\']'.format(ARGS.model_type))

###################################################################
## Train Helpers

def save_model(ARGS, type, model_dir, model, log_file_path, epoch):
    save_path = os.path.join(model_dir, type + '_' + str(epoch) + '.pth')
    print_and_log(log_file_path, "Saving Model path: " + str(save_path))
    torch.save(model.state_dict(), save_path) 
    if ARGS.model_save_latest_path:
        if not os.path.isdir(ARGS.model_save_latest_path):  # Create the latest saved pth directory if it doesn't exist
            os.makedirs(ARGS.model_save_latest_path)
        latest_save_path = os.path.join(ARGS.model_save_latest_path, type + '_' + 'latest' + '.pth')
        print_and_log(log_file_path, "Saving latest model: " + str(latest_save_path))
        torch.save(model.state_dict(), latest_save_path) 

# def removePercentTaggedFile(tag, model_dir):
#     for subdir, _, files in os.walk(model_dir):
#         for file in files:
#             if tag in file:
#                 filepath = os.path.join(subdir, file)
#                 os.remove(filepath)

def write_arguments_to_file(ARGS, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(ARGS)):
            f.write('%s: %s\n' % (key, str(value)))

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' +  e.strerror
  
    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' +  e.strerror
    
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('pytorch version: %s\n--------------------\n' % torch.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)

def print_and_log(log_file_path, string_to_write):
    print(string_to_write)
    with open(log_file_path, "a") as log_file:
        t = "[" + str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')) + "] " 
        log_file.write(t + string_to_write + "\n")

def schedule_lr(ARGS, log_file_path, optimizer, epoch):
    for lr_schedule_step in ARGS.lr_schedule_steps:
        if epoch == lr_schedule_step:
            for params in optimizer.param_groups:                 
                params['lr'] *= ARGS.lr_gamma
    print_and_log(log_file_path, "Learning rate: " + str(optimizer.param_groups[0]['lr']) + " Epoch: " + str(epoch))

###################################################################
## Features Helpers

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def crop_and_flip(image_data_rgb, for_dataloader=False):

    # image_data_rgb - should be shape=(160, 160, 3)  color_array=(255, 255, 255)

    image_BGR = cv2.cvtColor(image_data_rgb, cv2.COLOR_RGB2BGR)
    # resize image to [128, 128]
    resized = cv2.resize(image_BGR, (128, 128)) # (160, 160, 3) -> (128, 128, 3)

    # cv2.imwrite('aaaaaaa.png',image_data_rgb)
    # bp()

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    if not for_dataloader:
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    if not for_dataloader:
        flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)    
    return ccropped, flipped

def extract_norm_features(ccropped, flipped, model, device, tta = True):
    model.eval()
    with torch.no_grad():
        if tta:
            emb_batch = model(ccropped.to(device)).cpu() + model(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(model(ccropped.to(device)).cpu())
    return features


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        
        # dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        # norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        # similarity = dot / norm
        # dist = np.arccos(similarity) / math.pi

        dist = spatial.distance.cosine(embeddings1, embeddings2)

    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

###################################################################
## Train Helpers

def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn
