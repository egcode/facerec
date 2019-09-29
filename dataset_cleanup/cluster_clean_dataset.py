from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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




python3 dataset_cleanup/cluster_clean_dataset.py \
--affinity cosine \
--linkage average \
--distance_threshold 0.7 \
--h5_name data/dataset.h5 \
--output_clean_dataset data/dataset_clean \
--output_failed_images data/dataset_failed
'''

import os
import sys
import h5py
import numpy as np
import argparse
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
from datetime import datetime, timedelta

from shutil import copyfile

from pdb import set_trace as bp

def main(ARGS):

    if not os.path.isfile(ARGS.h5_name):
        assert "h5 file is not exist"

    out_dir = os.path.join(os.path.expanduser(ARGS.output_clean_dataset))
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)

    start_time = time.time()

    # Data for each person
    with h5py.File(ARGS.h5_name, 'r') as f:
        for person in f.keys():
            print("\npersonName: " + str(person))
    #         print("personLabel: " + str(f[person].attrs['label']))

            nrof_images = len(f[person].keys())
            embedding_size = 512
            embeddings_array = np.zeros((nrof_images, embedding_size))
            label_array = np.zeros((0,0))
            label_strings_array = []
            image_paths_array = []

            # print("\tembedding array shape: " + str(embeddings_array.shape))
            # print("\tnumber of images: " + str(nrof_images) + "  embedding size: " + str(embedding_size))

            for i, subgroup in enumerate(f[person].keys()):
                # print("\tlabel: " + str(i))
                embeddings_array[i, :] = f[person][subgroup]['embedding'][:]
                label_array = np.append(label_array, i)
                label_strings_array.append(str(subgroup))
                image_paths_array.append(f[person][subgroup].attrs['file_path'].decode('UTF-8'))

                # print("\tsubgroup: " + str(subgroup))
                # print("\t\tembedding data shape: " + str(f[person][subgroup]['embedding'][:].shape))

                # print("\t\tembedding data: " + str(f[person][subgroup]['embedding'][:4]))
                # print("\t\tpath data: " + str(f[person][subgroup].attrs['file_path']))

            # plt.figure(figsize=(10, 7))
            # plt.title(str(person))
            # dend = shc.dendrogram(shc.linkage(embeddings_array, method='average'),labels=label_strings_array,color_threshold=1.0)
            # plt.show()

            ### If Only One image in folder
            if nrof_images == 1:
                print("Folder contains only one file, don't do clustering")
                image_dir = os.path.join(out_dir, person)
                if not os.path.isdir(image_dir):  # Create the out directory if it doesn't exist
                    os.makedirs(image_dir)
                image_out = os.path.join(image_dir, label_strings_array[0])
                copyfile(image_paths_array[0], image_out)
                print("\tCopy Image image to path: " + str(image_out))
                print("Continue next person...")
                continue

            cluster = AgglomerativeClustering(n_clusters=None,
                                                affinity=ARGS.affinity, 
                                                linkage=ARGS.linkage,
                                                compute_full_tree=True,
                                                distance_threshold=ARGS.distance_threshold)
            pred = cluster.fit_predict(embeddings_array)
            print("CLUSTER PRED: " + str(pred))
            print("cluster pred shape: " + str(pred.shape))
            print("LABELS: " + str(np.array(label_strings_array)))
            print("label shape: " + str(len(label_strings_array)))


            uniq_labels, uniq_count = np.unique(pred, return_counts=True)
            # print("unique labels: " + str(uniq_labels) + "    " + "unique count: " + str(uniq_count))
            print("most often unique label: " + str(uniq_labels[0]) + "  we will only save this label from cluster")

            good_values = np.isin(pred, np.argmax(uniq_count))
            print("values to export: " + str(good_values))
            
            for i, image_path in enumerate(image_paths_array):
                if good_values[i] == True:
                    # print("\tExporting image: " + str(image_path))

                    image_dir = os.path.join(out_dir, person)
                    if not os.path.isdir(image_dir):  # Create the out directory if it doesn't exist
                        os.makedirs(image_dir)
                    image_out = os.path.join(image_dir, label_strings_array[i])
                    copyfile(image_paths_array[i], image_out)
                    print("\tCopy image to path: " + str(image_out))
                else:
                    if ARGS.output_failed_images != None:
                        failed_dir = os.path.join(os.path.expanduser(ARGS.output_failed_images))
                        if not os.path.isdir(failed_dir):  # Create the out directory if it doesn't exist
                            os.makedirs(failed_dir)

                        image_dir = os.path.join(failed_dir, person)
                        if not os.path.isdir(image_dir):  # Create the out directory if it doesn't exist
                            os.makedirs(image_dir)
                        image_out = os.path.join(image_dir, label_strings_array[i])
                        copyfile(image_paths_array[i], image_out)
                        print("\tCopy Failed image to path: " + str(image_out))
                        
    total_time = timedelta(seconds=int(time.time() - start_time))
    print(60*"=")
    print('All done. Total time: ' + str(total_time))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--affinity', type=str, help='Affinity type for clustering', default='cosine') # [cosine, euclidean]
    parser.add_argument('--linkage', type=str, help='Lingate method', default='average') #  [ward, complete, average, single]
    parser.add_argument('--distance_threshold', type=float, help='Dustance to cutoff embeddings', default=0.7)
    parser.add_argument('--h5_name', type=str, help='h5 file name', default='data/dataset.h5')
    parser.add_argument('--output_clean_dataset', type=str, help='Dir where to save clean dataset', default='data/dataset_clean')
    parser.add_argument('--output_failed_images', type=str, help='Dir where to save failed images', default=None)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
