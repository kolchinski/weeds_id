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
from util import train_model

np.set_printoptions(suppress=True)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

tiny_run = True

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_class = models.resnet18 if tiny_run else models.resnet101
model = model_class(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

last_layer_in_size = model.fc.in_features
model.fc = nn.Linear(last_layer_in_size, constants.NUM_CLASSES)

model.to(device)

params_to_train = [p for p in model.parameters() if p.requires_grad]
opt = optim.Adam(params_to_train, lr=constants.DEFAULT_INIT_LR)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=True) #TODO: parametrize, try cos
print([p.shape for p in params_to_train])

criterion = nn.CrossEntropyLoss()

dataloaders = data_util.get_dataloaders(batch_size=constants.DEFAULT_BATCH_SIZE,
                                        augment_data=False,
                                        max_num_pts = 40 if tiny_run else None)

# Train and evaluate
model, hist = train_model(model, dataloaders, criterion,
                          lr_scheduler, constants.MAX_EPOCHS, device, constants.NUM_CLASSES)
