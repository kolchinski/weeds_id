from __future__ import print_function
from __future__ import division
import argparse
import os
from datetime import datetime

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import data_util
import constants
from models.finetuned_resnet import FinetunedResnet

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Tell apart weeds with computer vision')
parser.add_argument("--init_lr", default=.01, type=float, help="Which learning rate to start with")
parser.add_argument("--max_epochs", default=100, type=int, help="How many epochs to run for")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size")
parser.add_argument("--save_every", default=5, type=int, help="Save every _this many_ epochs")
parser.add_argument("--max_num_pts", default=None, type=int, help="Reduce the train and val set to this many points")
parser.add_argument("--model", default='resnet50', type=str, help="Which model to train on the data")
parser.add_argument("--augment_data", action='store_true', help="Perform data augmentation on the training set?")
parser.add_argument("--freeze_resnet", action='store_true', help="Freeze parameters of pretrained model?")


def main():
    log_dir = os.path.join(
        os.getcwd(),
        'logs',
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir)

    print_log_file = open(os.path.join(log_dir, 'print_log.txt'), 'w', buffering=1)
    loss_log_file = open(os.path.join(log_dir, 'loss_and_accuracy_log.txt'), 'w', buffering=1)

    print("PyTorch Version: ",torch.__version__, file=print_log_file)
    print("Torchvision Version: {}\n\n".format(torchvision.__version__), file=print_log_file)

    args = parser.parse_args()
    for k,v in vars(args).items():
        print(k,v, file=print_log_file)
    print('\n\n', file=print_log_file)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet50': model_class = models.resnet50
    elif args.model == 'resnet18': model_class = models.resnet18
    else: raise Exception('Invalid model name')

    dataloaders = data_util.get_dataloaders(batch_size=args.batch_size,
                                            augment_data=args.augment_data,
                                            max_num_pts=args.max_num_pts)

    resnet = FinetunedResnet(model_class, args.freeze_resnet, constants.NUM_CLASSES, device)
    resnet.train(dataloaders, args.init_lr, args.max_epochs, args.save_every,
                 log_dir, print_log_file, loss_log_file)

    loss_log_file.close()
    print_log_file.close()



if __name__ == '__main__':
    main()

