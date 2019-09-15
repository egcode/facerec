from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import argparse
from evaluate_helpers import *

from models.resnet import *
from models.irse import *

from pdb import set_trace as bp

"""
EXAMPLE:
python3 evaluate.py  \
--model_path ./pth/IR_50_MODEL_arcface_casia_epoch56_lfw9925.pth \
--model_type IR_50 \
--num_workers 8 \
--batch_size 100
"""

class EvaluateDataset(data.Dataset):
    
    def __init__(self, paths, actual_issame, input_size):
    
        self.paths = paths
        self.actual_issame = actual_issame

        self.nrof_embeddings = len(self.actual_issame)*2  # nrof_pairs * nrof_images_per_pair
        self.labels_array = np.arange(0,self.nrof_embeddings)

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        self.transforms = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path)
        data = img.convert('RGB')
        data = self.transforms(data)
        label = self.labels_array[index]
        return data.float(), label

    def __len__(self):
        return len(self.paths)


def evaluate_forward_pass(model, lfw_loader, lfw_dataset, embedding_size, device, lfw_nrof_folds, distance_metric, subtract_mean):

    nrof_images = lfw_dataset.nrof_embeddings

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    with torch.no_grad():
        for i, (data, label) in enumerate(lfw_loader):

            data, label = data.to(device), label.to(device)

            feats = model(data)
            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            lab_array[lab] = lab
            emb_array[lab, :] = emb

            if i % 10 == 9:
                print('.', end='')
                sys.stdout.flush()
        print('')
    embeddings = emb_array

    # np.save('embeddings.npy', embeddings) 
    # embeddings = np.load('embeddings.npy')

    # np.save('embeddings_casia.npy', embeddings) 
    # embeddings = np.load('embeddings_casia.npy')

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, lfw_dataset.actual_issame, nrof_folds=lfw_nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return tpr, fpr, accuracy, val, val_std, far

#-------------------------------------------------------------
# LFW
def get_paths_issame_LFW(lfw_dir):

    lfw_images_dir = lfw_dir + '/images'
    lfw_pairs = lfw_dir + '/pairs_LFW.txt'

    # Read the file containing the pairs used for testing
    pairs = read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = get_paths(os.path.expanduser(lfw_images_dir), pairs)

    return paths, actual_issame

#-------------------------------------------------------------

# CPLFW
def get_paths_issame_CPLFW(cplfw_dir):
    cplfw_images_dir = cplfw_dir + '/images'
    cplfw_pairs = cplfw_dir + '/pairs_CPLFW.txt'
    return get_paths_issame_ca_or_cp_lfw(cplfw_images_dir, cplfw_pairs)

# CALFW
def get_paths_issame_CALFW(calfw_dir):
    calfw_images_dir = calfw_dir + '/images'
    calfw_pairs = calfw_dir + '/pairs_CALFW.txt'
    return get_paths_issame_ca_or_cp_lfw(calfw_images_dir, calfw_pairs)


def get_paths_issame_ca_or_cp_lfw(lfw_dir, lfw_pairs):

    pairs = []
    with open(lfw_pairs, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    arr = np.array(pairs)

    paths = []
    actual_issame = []
    for count, person in enumerate(arr, 1): # Start counting from 1
        if count % 2 == 0:
            first_in_pair = arr[count-2]
            second_in_pair = person

            dir = os.path.expanduser(lfw_dir)
            path1 = os.path.join(dir, first_in_pair[0])
            path2 = os.path.join(dir, second_in_pair[0])
            paths.append(path1)
            paths.append(path2)

            if first_in_pair[1] != '0':
                actual_issame.append(True)
            else:
                actual_issame.append(False)
    
    return paths, actual_issame

#-------------------------------------------------------------
# CFP_FF and CFP_FP
def get_paths_issame_CFP(cfp_dir, type='FF'):

    pairs_list_F = cfp_dir + '/Pair_list_F.txt'
    pairs_list_P = cfp_dir + '/Pair_list_P.txt'

    path_hash_F = {}
    with open(pairs_list_F, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_F[pair[0]] = cfp_dir + '/' + pair[1]

    path_hash_P = {}
    with open(pairs_list_P, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            path_hash_P[pair[0]] = cfp_dir + '/' + pair[1]


    paths = []
    actual_issame = []

    if type == 'FF':
        root_FF_or_FP = cfp_dir + '/Split/FF'
    else:
        root_FF_or_FP = cfp_dir + '/Split/FP'


    for subdir, _, files in os.walk(root_FF_or_FP):
        for file in files:
            filepath = os.path.join(subdir, file)

            pairs_arr = parse_dif_same_file(filepath)
            for pair in pairs_arr:
            
                first = path_hash_F[pair[0]]

                if type == 'FF':
                    second = path_hash_F[pair[1]]
                else:
                    second = path_hash_P[pair[1]]
                

                paths.append(first)
                paths.append(second)

                if file == 'diff.txt':
                    actual_issame.append(False)
                else:
                    actual_issame.append(True)

    return paths, actual_issame


def parse_dif_same_file(filepath):
    pairs_arr = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split(',')
            pairs_arr.append(pair)
    return pairs_arr     

#-------------------------------------------------------------

def get_evaluate_dataset_and_loader(root_dir, type='LFW', num_workers=2, input_size=[112, 112], batch_size=100):
    ######## dataset setup
    if type == 'CALFW':
        paths, actual_issame = get_paths_issame_CALFW(root_dir)
    elif type == 'CPLFW':
        paths, actual_issame = get_paths_issame_CPLFW(root_dir)
    elif type == 'CFP_FF':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FF')
    elif type == 'CFP_FP':
        paths, actual_issame = get_paths_issame_CFP(root_dir, type='FP')
    else:
        paths, actual_issame = get_paths_issame_LFW(root_dir)

    dataset = EvaluateDataset(paths=paths, actual_issame=actual_issame, input_size=input_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataset, loader


def print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far):
    print("=" * 60)
    print("Validation TYPE: {}".format(type))
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    # print('Equal Error Rate (EER): %1.3f' % eer)
    print("=" * 60)



def main(ARGS):

    if ARGS.model_path == None:
        raise AssertionError("Path should not be None")

    ######### distance_metric = 1 #### if CenterLoss = 0, If Cosface = 1
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    print('Model type: %s' % ARGS.model_type)
    if ARGS.model_type == 'ResNet_50':
        model = ResNet_50(ARGS.input_size)
    elif ARGS.model_type == 'ResNet_101':
        model = ResNet_101(ARGS.input_size)
    elif ARGS.model_type == 'ResNet_152':
        model = ResNet_152(ARGS.input_size)
    elif ARGS.model_type == 'IR_50':
        model = IR_50(ARGS.input_size)
    elif ARGS.model_type == 'IR_101':
        model = IR_101(ARGS.input_size)
    elif ARGS.model_type == 'IR_152':
        model = IR_152(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_50':
        model = IR_SE_50(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_101':
        model = IR_SE_101(ARGS.input_size)
    elif ARGS.model_type == 'IR_SE_152':
        model = IR_SE_152(ARGS.input_size)
    else:
        raise AssertionError('Unsuported model_type {}. We only support: [\'ResNet_50\', \'ResNet_101\', \'ResNet_152\', \'IR_50\', \'IR_101\', \'IR_152\', \'IR_SE_50\', \'IR_SE_101\', \'IR_SE_152\']'.format(ARGS.model_type))

    if use_cuda:
        model.load_state_dict(torch.load(ARGS.model_path))
    else:
        model.load_state_dict(torch.load(ARGS.model_path, map_location='cpu'))

    model.to(device)
    embedding_size = 512
    model.eval()

    ##########################################################################################
    #### Evaluate LFW Example
    type='LFW'
    root_dir='./data/lfw_112'
    dataset, loader = get_evaluate_dataset_and_loader(root_dir=root_dir, 
                                                            type=type, 
                                                            num_workers=ARGS.num_workers, 
                                                            input_size=[112, 112], 
                                                            batch_size=ARGS.batch_size)

    print('Runnning forward pass on {} images'.format(type))

    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                loader, 
                                                                dataset, 
                                                                embedding_size, 
                                                                device,
                                                                lfw_nrof_folds=10, 
                                                                distance_metric=1, 
                                                                subtract_mean=False)

    print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far)
    #### End of Evaluate LFW Example
    ##########################################################################################


    ##########################################################################################
    ### Evaluate CALFW Example
    type='CALFW'
    root_dir='./data/calfw_112'
    dataset, loader = get_evaluate_dataset_and_loader(root_dir=root_dir, 
                                                            type=type, 
                                                            num_workers=ARGS.num_workers, 
                                                            input_size=[112, 112], 
                                                            batch_size=ARGS.batch_size)

    print('Runnning forward pass on {} images'.format(type))

    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                loader, 
                                                                dataset, 
                                                                embedding_size, 
                                                                device,
                                                                lfw_nrof_folds=10, 
                                                                distance_metric=1, 
                                                                subtract_mean=False)

    print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far)
    #### End of Evaluate CALFW Example
    ##########################################################################################

    ##########################################################################################
    ### Evaluate CPLFW Example
    type='CPLFW'
    root_dir='./data/cplfw_112'
    dataset, loader = get_evaluate_dataset_and_loader(root_dir=root_dir, 
                                                            type=type, 
                                                            num_workers=ARGS.num_workers, 
                                                            input_size=[112, 112], 
                                                            batch_size=ARGS.batch_size)

    print('Runnning forward pass on {} images'.format(type))

    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                loader, 
                                                                dataset, 
                                                                embedding_size, 
                                                                device,
                                                                lfw_nrof_folds=10, 
                                                                distance_metric=1, 
                                                                subtract_mean=False)

    print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far)
    #### End of Evaluate CPLFW Example
    ##########################################################################################


    ##########################################################################################
    ### Evaluate CFP_FF Example
    type='CFP_FF'
    root_dir='./data/cfp_112'
    dataset, loader = get_evaluate_dataset_and_loader(root_dir=root_dir, 
                                                            type=type, 
                                                            num_workers=ARGS.num_workers, 
                                                            input_size=[112, 112], 
                                                            batch_size=ARGS.batch_size)

    print('Runnning forward pass on {} images'.format(type))

    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                loader, 
                                                                dataset, 
                                                                embedding_size, 
                                                                device,
                                                                lfw_nrof_folds=10, 
                                                                distance_metric=1, 
                                                                subtract_mean=False)

    print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far)
    #### End of Evaluate CFP_FF Example
    ##########################################################################################


    ##########################################################################################
    ### Evaluate CFP_FP Example
    type='CFP_FP'
    root_dir='./data/cfp_112'
    dataset, loader = get_evaluate_dataset_and_loader(root_dir=root_dir, 
                                                            type=type, 
                                                            num_workers=ARGS.num_workers, 
                                                            input_size=[112, 112], 
                                                            batch_size=ARGS.batch_size)

    print('Runnning forward pass on {} images'.format(type))

    tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                loader, 
                                                                dataset, 
                                                                embedding_size, 
                                                                device,
                                                                lfw_nrof_folds=10, 
                                                                distance_metric=1, 
                                                                subtract_mean=False)

    print_evaluate_result(type, tpr, fpr, accuracy, val, val_std, far)
    #### End of Evaluate CFP_FP Example
    ##########################################################################################


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Model weights.', default=None)
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    parser.add_argument('--batch_size', type=int, help='Number of batches while validating model.', default=100)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
