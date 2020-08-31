from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import protobuf.face_dataset_pb2 as face_dataset
import h5py                                                                                                                                                                                   

"""
EXAMPLE:
python3 protobuf/convert_from_h5_to_protobuf.py  \
--source_h5 ./data/out_embeddings/dataset_targarien.h5 \
--target_protobuf ./data/out_embeddings/dataset_targarien.protobuf


python3 protobuf/convert_from_h5_to_protobuf.py  \
--source_h5 ./data/out_embeddings/dataset_golovan.h5 \
--target_protobuf ./data/out_embeddings/dataset_golovan.protobuf
"""

def main(ARGS):

    if ARGS.source_h5 == None:
        raise AssertionError("source_h5 path should not be None")

    if ARGS.target_protobuf == None:
        raise AssertionError("target_protobuf path should not be None")

    dataset = face_dataset.DatasetObject()
    tempDatasetArr = []

    with h5py.File(ARGS.source_h5, 'r') as f:
        for person in f.keys():
            embedding_premade = f[person]['embedding'][:]

            # print("Name: " + person)
            # print("Embedding: " + str(embedding_premade))

            face = face_dataset.FaceObject()
            face.name = person
            face.embeddings.extend(embedding_premade) 

            tempDatasetArr.append(face)

    dataset.faceobjects.extend(tempDatasetArr)


    print(60*"=" + "Print Dataset")
    print(dataset)
    print(60*"=")

    print(60*"=" + "Print Dataset as a String")
    print(dataset.SerializeToString())
    print(60*"=")


    print(60*"=" + "Loop")
    for descriptor in dataset.DESCRIPTOR.fields:
        extractedFaces = getattr(dataset, descriptor.name)
        for i in range(len(extractedFaces)): 
            print("\nFace: " + str(i) + " " + extractedFaces[i].name)
            print("Embedding: " + str(extractedFaces[i].embeddings[:3]))
    print(60*"=")


    # Write the new address book back to disk.
    f = open(ARGS.target_protobuf, "wb")
    f.write(dataset.SerializeToString())
    f.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_h5', type=str, help='Source hdf5 dataset file.', default=None)
    parser.add_argument('--target_protobuf', type=str, help='Target Protobuf dataset file.', default=None)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
