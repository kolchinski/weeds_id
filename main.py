#Code adapted from PyTorch fine tuning tutorial at
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import data_util
import constants
from models.finetuned_resnet import FinetunedResnet

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Tell apart weeds with computer vision')
parser.add_argument("--init_lr", default=.01, type=float, help="Which learning rate to start with")
parser.add_argument("--max_epochs", default=1000, type=int, help="How many epochs to run for")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size")
parser.add_argument("--max_num_pts", default=None, type=int, help="Reduce the train and val set to this many points")
parser.add_argument("--model", default='resnet50', type=str, help="Which model to train on the data")
parser.add_argument("--augment_data", action='store_true', help="Perform data augmentation on the training set?")


def main():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: {}\n\n".format(torchvision.__version__))

    args = parser.parse_args()
    for k,v in vars(args).items():
        print(k,v)
    print('\n\n')

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'resnet50': model_class = models.resnet50
    elif args.model == 'resnet18': model_class = models.resnet18
    else: raise Exception('Invalid model name')

    dataloaders = data_util.get_dataloaders(batch_size=args.batch_size,
                                            augment_data=args.augment_data,
                                            max_num_pts=args.max_num_pts)

    resnet = FinetunedResnet(model_class, constants.NUM_CLASSES, device)
    resnet.train(dataloaders, args.init_lr, args.max_epochs)



if __name__ == '__main__':
    main()

