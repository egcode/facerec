from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
import argparse
from sklearn import metrics
from losses.ArcFaceLossMargin import ArcFaceLossMargin, ArcFaceLossMargin2
from losses.CosFaceLossMargin import CosFaceLossMargin
from losses.CombinedLossMargin import CombinedLossMargin
from losses.CenterLoss import CenterLoss
from losses.FocalLoss import FocalLoss
from dataset.get_data import get_data
from models.resnet import *
from models.irse import *
from helpers import *
from evaluate import *
from datetime import datetime, timedelta
import time
from logger import Logger
from pdb import set_trace as bp

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


'''
EXAMPLE:
python3 train.py \
--model_type IR_50 \
--data_dir ./data/CASIA_Webface_160 \
--batch_size 128 \
--batch_size_test 128 \
--evaluate_batch_size 128 \
--criterion_type arcface \
--total_loss_type softmax \
--optimizer_type sgd_bn \
--margin_s 32.0 \
--margin_m 0.5 \
--validation_set_split_ratio 0.0 \
--lr 0.1 \
--lr_schedule_steps 30 55 75 \
--apex_opt_level 2 

'''


def train(ARGS, model, device, train_loader, total_loss, loss_criterion, optimizer, log_file_path, model_dir, logger, epoch):
    model.train()
    t = time.time()
    log_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        tt = time.time()

        data, target = data.to(device), target.to(device)

        # Forward prop.
        features = model(data)

        if ARGS.criterion_type == 'arcface':
            logits = loss_criterion(features, target)
            loss = total_loss(logits, target)
        # elif ARGS.criterion_type == 'arcface2':
        #     logits = loss_criterion(features, target)
        #     loss = total_loss(logits, target)
        elif ARGS.criterion_type == 'cosface':
            logits, mlogits = loss_criterion(features, target)
            loss = total_loss(mlogits, target)
        elif ARGS.criterion_type == 'combined':
            logits = loss_criterion(features, target)
            loss = total_loss(logits, target)
        elif ARGS.criterion_type == 'centerloss':
            weight_cent = 1.
            loss_cent, outputs = loss_criterion(features, target)
            loss_cent *= weight_cent
            los_softm = total_loss(outputs, target)
            loss = los_softm + loss_cent

        optimizer.zero_grad()

        if APEX_AVAILABLE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if ARGS.criterion_type == 'centerloss':
            # by doing so, weight_cent would not impact on the learning of centers
            for param in loss_criterion.parameters():
                param.grad.data *= (1. / weight_cent)

        # Update weights
        optimizer.step()

        time_for_batch = int(time.time() - tt)
        time_for_current_epoch = int(time.time() - t)
        percent = 100. * batch_idx / len(train_loader)

        log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tbatch_time: {}   Total time for epoch: {}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            percent, loss.item(), timedelta(seconds=time_for_batch), timedelta(seconds=time_for_current_epoch))
        # print_and_log(log_file_path, log)
        print(log)

        log_loss = loss.item()

        # loss_epoch_and_percent - last two digits - Percent of epoch completed
        logger.scalar_summary("loss_epoch_and_percent", log_loss, (epoch*100)+(100. * batch_idx / len(train_loader)))

    logger.scalar_summary("loss", log_loss, epoch)

    time_for_epoch = int(time.time() - t)
    print_and_log(log_file_path, 'Total time for epoch: {}'.format(timedelta(seconds=time_for_epoch)))

    if epoch % ARGS.model_save_interval == 0 or epoch == ARGS.epochs:
        save_model(ARGS, ARGS.model_type, model_dir, model, log_file_path, epoch)
        save_model(ARGS, ARGS.criterion_type, model_dir, loss_criterion, log_file_path, epoch)

def test(ARGS, model, device, test_loader, total_loss, loss_criterion, log_file_path, logger, epoch):
    if test_loader == None:
        return
    model.eval()
    correct = 0
    if epoch % ARGS.test_interval == 0 or epoch == ARGS.epochs:
        model.eval()
        t = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                feats = model(data)

                if ARGS.criterion_type == 'arcface':
                    logits = loss_criterion(feats, target)
                    outputs = logits
                # elif ARGS.criterion_type == 'arcface2':
                #     logits = loss_criterion(feats, target)
                #     outputs = logits
                elif ARGS.criterion_type == 'cosface':
                    logits, _ = loss_criterion(feats, target)
                    outputs = logits
                elif ARGS.criterion_type == 'combined':
                    logits = loss_criterion(feats, target)
                    outputs = logits
                elif ARGS.criterion_type == 'centerloss':
                    _, outputs = loss_criterion(feats, target)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target.data).sum()
                

        accuracy = 100. * correct / len(test_loader.dataset)
        log = '\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy)
        print_and_log(log_file_path, log)

        logger.scalar_summary("accuracy", accuracy, epoch)

        time_for_test = int(time.time() - t)
        print_and_log(log_file_path, 'Total time for test: {}'.format(timedelta(seconds=time_for_test)))


def evaluate(ARGS, validation_data_dic, model, device, log_file_path, logger, distance_metric, epoch):
    if epoch % ARGS.evaluate_interval == 0 or epoch == ARGS.epochs:

        embedding_size = ARGS.features_dim

        for val_type in ARGS.validations:
            dataset = validation_data_dic[val_type+'_dataset']
            loader = validation_data_dic[val_type+'_loader']

            model.eval()
            t = time.time()
            print('\n\nRunnning forward pass on {} images'.format(val_type))

            tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                        loader, 
                                                                        dataset, 
                                                                        embedding_size, 
                                                                        device,
                                                                        lfw_nrof_folds=ARGS.evaluate_nrof_folds, 
                                                                        distance_metric=distance_metric, 
                                                                        subtract_mean=ARGS.evaluate_subtract_mean)


            print_and_log(log_file_path, '\nEpoch: '+str(epoch))
            print_and_log(log_file_path, 'Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print_and_log(log_file_path, 'Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print_and_log(log_file_path, 'Area Under Curve (AUC): %1.3f' % auc)

            # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            # print('Equal Error Rate (EER): %1.3f' % eer)
            time_for_val = int(time.time() - t)
            print_and_log(log_file_path, 'Total time for {} evaluation: {}'.format(val_type, timedelta(seconds=time_for_val)))
            print("\n")
                
            logger.scalar_summary(val_type +"_accuracy", np.mean(accuracy), epoch)


def main(ARGS):

    # Dirs
    subdir = datetime.strftime(datetime.now(), '%Y-%m-%d___%H-%M-%S')
    out_dir = os.path.join(os.path.expanduser(ARGS.out_dir), subdir)
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)
    model_dir = os.path.join(os.path.expanduser(out_dir), 'model')
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    tensorboard_dir = os.path.join(os.path.expanduser(out_dir), 'tensorboard')
    if not os.path.isdir(tensorboard_dir):  # Create the tensorboard directory if it doesn't exist
        os.makedirs(tensorboard_dir)

    # Write arguments to a text file
    write_arguments_to_file(ARGS, os.path.join(out_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    store_revision_info(src_path, out_dir, ' '.join(sys.argv))

    log_file_path = os.path.join(out_dir, 'training_log.txt')
    logger = Logger(tensorboard_dir)

    ################### Pytorch: ###################
    print_and_log(log_file_path, "Pytorch version:  " + str(torch.__version__))
    use_cuda = torch.cuda.is_available()
    print_and_log(log_file_path, "Use CUDA: " + str(use_cuda))
    print_and_log(log_file_path, "Cuda Version:  " + str(torch.version.cuda))
    print_and_log(log_file_path, "cudnn enabled:  " + str(torch.backends.cudnn.enabled))
    print_and_log(log_file_path, "Use APEX: " + str(APEX_AVAILABLE))
    if APEX_AVAILABLE:
            print_and_log(log_file_path, "APEX level: " + str(ARGS.apex_opt_level))
    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    print('Data directory: %s' % ARGS.data_dir)
    train_loader, test_loader = get_data(ARGS, device)

    ######## Validation Data setup
    validation_paths_dic = {
                    "LFW" : ARGS.lfw_dir,
                    "CALFW" : ARGS.calfw_dir,
                    "CPLFW" : ARGS.cplfw_dir,
                    "CFP_FF" : ARGS.cfp_ff_dir,
                    "CFP_FP" : ARGS.cfp_fp_dir
                    }
    print_and_log(log_file_path, "Validation_paths_dic: " + str(validation_paths_dic))
    validation_data_dic = {}
    for val_type in ARGS.validations:
        print_and_log(log_file_path, 'Init dataset and loader for validation type: {}'.format(val_type))
        dataset, loader = get_evaluate_dataset_and_loader(root_dir=validation_paths_dic[val_type], 
                                                                type=val_type, 
                                                                num_workers=ARGS.num_workers, 
                                                                input_size=ARGS.input_size, 
                                                                batch_size=ARGS.evaluate_batch_size)
        validation_data_dic[val_type+'_dataset'] = dataset
        validation_data_dic[val_type+'_loader'] = loader
    

    ####### Model setup
    print('Model type: %s' % ARGS.model_type)
    model = get_model(ARGS.model_type, ARGS.input_size)
    if ARGS.model_path != None:
        if use_cuda:
            model.load_state_dict(torch.load(ARGS.model_path))
        else:
            model.load_state_dict(torch.load(ARGS.model_path, map_location='cpu'))
    model = model.to(device)

    if ARGS.total_loss_type == 'softmax':
        total_loss = nn.CrossEntropyLoss().to(device)
    elif ARGS.total_loss_type == 'focal':
        total_loss = FocalLoss().to(device)
    else:
        raise AssertionError('Unsuported total_loss_type {}. We only support:  [\'softmax\', \'focal\']'.format(ARGS.total_loss_type))

    ####### Criterion setup
    print('Criterion type: %s' % ARGS.criterion_type)
    if ARGS.criterion_type == 'arcface':
        distance_metric = 1
        loss_criterion = ArcFaceLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m=ARGS.margin_m).to(device)
    # elif ARGS.criterion_type == 'arcface2':
    #     distance_metric = 1
    #     loss_criterion = ArcFaceLossMargin2(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m=ARGS.margin_m).to(device)
    elif ARGS.criterion_type == 'cosface':
        distance_metric = 1
        loss_criterion = CosFaceLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m=ARGS.margin_m).to(device)
    elif ARGS.criterion_type == 'combined':
        distance_metric = 1
        loss_criterion = CombinedLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, device=device, s=ARGS.margin_s, m1=ARGS.margin_m1, m2=ARGS.margin_m2).to(device)
    elif ARGS.criterion_type == 'centerloss':
        distance_metric = 0
        loss_criterion = CenterLoss(device=device, num_classes=train_loader.dataset.num_classes, feat_dim=ARGS.features_dim, use_gpu=use_cuda)
    else:
        raise AssertionError('Unsuported criterion_type {}. We only support:  [\'arcface\', \'cosface\', \'combined\', \'centerloss\']'.format(ARGS.criterion_type))

    if ARGS.loss_path != None:
        if use_cuda:
            loss_criterion.load_state_dict(torch.load(ARGS.loss_path))
        else:
            loss_criterion.load_state_dict(torch.load(ARGS.loss_path, map_location='cpu'))


    if ARGS.optimizer_type == 'sgd_bn':
        ##################
        if ARGS.model_type.find("IR") >= 0:
            model_params_only_bn, model_params_no_bn = separate_irse_bn_paras(
                model)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, loss_params_no_bn = separate_irse_bn_paras(loss_criterion)
        else:
            model_params_only_bn, model_params_no_bn = separate_resnet_bn_paras(
                model)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, loss_params_no_bn = separate_resnet_bn_paras(loss_criterion)

        optimizer = optim.SGD([{'params': model_params_no_bn + loss_params_no_bn, 'weight_decay': ARGS.weight_decay}, 
                            {'params': model_params_only_bn}], lr = ARGS.lr, momentum = ARGS.momentum)

    elif ARGS.optimizer_type == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': loss_criterion.parameters()}],
                                         lr=ARGS.lr, betas=(ARGS.beta1, 0.999))
    elif ARGS.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': loss_criterion.parameters()}],
                                        lr=ARGS.lr, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay)
    else:
        raise AssertionError('Unsuported optimizer_type {}. We only support:  [\'sgd_bn\',\'adam\',\'sgd\']'.format(ARGS.optimizer_type))


    if APEX_AVAILABLE:
        if ARGS.apex_opt_level==0:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O0", loss_scale=1.0
            )
        elif ARGS.apex_opt_level==1:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O1", loss_scale="dynamic"
            )
        elif ARGS.apex_opt_level==2:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", 
                keep_batchnorm_fp32=True, loss_scale="dynamic"
            )
        else:
            raise AssertionError('Unsuported apex_opt_level {}. We only support:  [0, 1, 2]'.format(ARGS.apex_opt_level))


    #### Since StepLR and MultiStepLR are both buggy, use custom schedule_lr method
    # sheduler = lr_scheduler.StepLR(optimizer, ARGS.lr_step, gamma=ARGS.lr_gamma)
    # sheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=ARGS.lr_gamma)

    for epoch in range(1, ARGS.epochs + 1):
        schedule_lr(ARGS, log_file_path, optimizer, epoch)
        logger.scalar_summary("lr", optimizer.param_groups[0]['lr'], epoch)

        train(ARGS, model, device, train_loader, total_loss, loss_criterion, optimizer, log_file_path, model_dir, logger, epoch)
        test(ARGS, model, device, test_loader, total_loss, loss_criterion, log_file_path, logger, epoch)
        evaluate(ARGS, validation_data_dic, model, device, log_file_path, logger, distance_metric, epoch)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # Out    
    parser.add_argument('--out_dir', type=str,  help='Directory where to trained models and event logs.', default='./out')
    # Training
    parser.add_argument('--epochs', type=int, help='Training epochs training.', default=125)
    # Data
    parser.add_argument('--input_size', type=str, help='support: [112, 112] and [224, 224]', default=[112, 112])
    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/CASIA-WebFace_160')
    parser.add_argument('--num_workers', type=int, help='Number of threads to use for data pipeline.', default=8)
    parser.add_argument('--batch_size', type=int, help='Number of batches while training model.', default=512)
    parser.add_argument('--batch_size_test', type=int, help='Number of batches while testing model.', default=512)
    parser.add_argument('--validation_set_split_ratio', type=float, help='The ratio of the total dataset to use for validation', default=0.01)
    parser.add_argument('--min_nrof_val_images_per_class', type=float, help='Classes with fewer images will be removed from the validation set', default=0)
    # Model
    parser.add_argument('--model_path', type=str, help='Model weights if needed.', default=None)
    parser.add_argument('--model_type', type=str, help='Model type to use for training.', default='ResNet_50')# support: ['LightNet', 'MobileNet', 'ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--features_dim', type=int, help='Number of features for loss.', default=512)
    # Optimizer
    parser.add_argument('--optimizer_type', type=str, help='Optimizer Type.', default='sgd_bn') # support: ['sgd_bn','adam','sgd']
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--lr_schedule_steps', nargs='+', type=int, help='Steps when to multiply lr by lr_gamma.', default=[35, 65, 85])
    parser.add_argument('--lr_gamma', type=float, help='Every step lr will be multiplied by this value.', default=0.1)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # Loss 
    parser.add_argument('--total_loss_type', type=str, help='type of loss cosface or centerloss.', default='softmax') # support ['softmax', 'focal']
    parser.add_argument('--criterion_type', type=str, help='type of loss cosface or centerloss.', default='arcface') # support ['arcface', 'cosface', 'combined', 'centerloss']
    parser.add_argument('--loss_path', type=str, help='Loss weights if needed.', default=None)
    parser.add_argument('--margin_s', type=float, help='scale for feature.', default=32.0)
    parser.add_argument('--margin_m', type=float, help='margin for arcface loss.', default=0.5)
    parser.add_argument('--margin_m1', type=float, help='combined margin m1.', default=0.2)
    parser.add_argument('--margin_m2', type=float, help='combined margin m2.', default=0.35)
    parser.add_argument('--apex_opt_level', type=int, help='Apex opt level, 0=None,1=half,2=full.', default=2)    
    # Intervals
    parser.add_argument('--model_save_interval', type=int, help='Save model with every interval epochs.', default=1)
    parser.add_argument('--model_save_latest_path', type=str, help='Save latest saved model path.', default=None)
    parser.add_argument('--test_interval', type=int, help='Perform test with every interval epochs.', default=1)
    parser.add_argument('--evaluate_interval', type=int, help='Perform validation test with every interval epochs.', default=1)    
    # Validation
    parser.add_argument('--validations', nargs='+', help='Face validation types', default=['LFW', 'CALFW', 'CPLFW', 'CFP_FF', 'CFP_FP'])  # support ['LFW', 'CALFW', 'CPLFW', 'CFP_FF', 'CFP_FP']
    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/lfw_112')
    parser.add_argument('--calfw_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/calfw_112')
    parser.add_argument('--cplfw_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/cplfw_112')
    parser.add_argument('--cfp_ff_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/cfp_112')
    parser.add_argument('--cfp_fp_dir', type=str, help='Path to the data directory containing aligned face patches.', default='./data/cfp_112')
    # parser.add_argument('--evaluate_distance_metric', type=int, help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--evaluate_subtract_mean', help='Subtract feature mean before calculating distance.', action='store_true', default=False)
    parser.add_argument('--evaluate_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--evaluate_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
