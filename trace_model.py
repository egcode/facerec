from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import torch
import torchvision
from helpers import get_model

"""
EXAMPLE:
python3 trace_model.py  \
--import_pytorch_model_path ./data/pth/MobileNet_V3_Small_40.pth \
--export_traced_model_path ./data/pth/MobileNet_V3_Small_40_traced_model.pt \
--model_type MobileNet_V3_Small
"""

def main(ARGS):

    if ARGS.import_pytorch_model_path == None:
        raise AssertionError("Path should not be None")

    ######### distance_metric = 1 #### if CenterLoss = 0, If Cosface = 1

    ####### Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####### Model setup
    print("Use CUDA: " + str(use_cuda))
    print('Model type: %s' % ARGS.model_type)
    model = get_model(ARGS.model_type, ARGS.input_size)

    if use_cuda:
        model.load_state_dict(torch.load(ARGS.import_pytorch_model_path))
    else:
        model.load_state_dict(torch.load(ARGS.import_pytorch_model_path, map_location='cpu'))

    model.to(device)
    embedding_size = 512
    model.eval()

    example = torch.rand(1, 3, 112, 112)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(ARGS.export_traced_model_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--import_pytorch_model_path', type=str, help='Trained model path.', default=None)
    parser.add_argument('--export_traced_model_path', type=str, help='Export Torchscript model.', default=None)
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='IR_50')# support: ['LightNet', 'MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large', 'ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
