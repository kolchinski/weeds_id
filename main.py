#Code adapted from PyTorch fine tuning tutorial at
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division

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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

tiny_run = True

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_class = models.resnet18 if tiny_run else models.resnet50


dataloaders = data_util.get_dataloaders(batch_size=constants.DEFAULT_BATCH_SIZE,
                                        augment_data=False,
                                        max_num_pts = 40 if tiny_run else None)

resnet = FinetunedResnet(model_class, constants.NUM_CLASSES, device)
resnet.train(dataloaders, constants.DEFAULT_INIT_LR, constants.MAX_EPOCHS)