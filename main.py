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
import matplotlib.pyplot as plt
import time
import os
import copy

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

num_classes = 9
input_size = 224


model = models.resnet18(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

last_layer_in_size = model.fc.in_features
model.fc = nn.Linear(last_layer_in_size, num_classes)

print(model)