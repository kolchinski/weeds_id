#Code adapted from PyTorch fine tuning tutorial at
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

np.set_printoptions(suppress=True)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

num_classes = 9
input_size = 224
data_dir = './deep_weeds_dataset'
num_epochs = 1000
tiny_run = True
batch_size = 8 if tiny_run else 1024

def train_model(model, dataloaders, criterion, optim_scheduler, num_epochs, device):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            confusion_matrix = np.zeros((num_classes, num_classes))

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optim_scheduler.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        lr_scheduler.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for label, pred in zip(labels.data, preds):
                    confusion_matrix[label, pred] += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            for i in range(num_classes):
                print('Class {} ({}): Accuracy {}/{} ({}%)'.format(
                    i, dataloaders[phase].dataset.dataset.classes[i], confusion_matrix[i,i],
                    confusion_matrix[i].sum(), 100.*confusion_matrix[i,i]/confusion_matrix[i].sum()
                ))
            print(confusion_matrix)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                lr_scheduler.step(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=0.1),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), shear=(-30,30)),
        transforms.RandomResizedCrop(input_size, scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(
    os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

if tiny_run:
    train_indices = np.random.choice(len(image_datasets['train']), 40, replace=False)
    val_indices = np.random.choice(len(image_datasets['val']), 40, replace=False)
else:
    train_indices = range(len(image_datasets['train']))
    val_indices = range(len(image_datasets['val']))

train_set = Subset(image_datasets['train'], train_indices)
val_set   = Subset(image_datasets['val'], val_indices)
sets =  {'train' : train_set, 'val' : val_set}

# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader(
    sets[x], batch_size=batch_size, shuffle=True, num_workers=4)
    for x in ['train', 'val']}


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_class = models.resnet18 if tiny_run else models.resnet101
model = model_class(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

last_layer_in_size = model.fc.in_features
model.fc = nn.Linear(last_layer_in_size, num_classes)

model.to(device)

print(model)

params_to_train = [p for p in model.parameters() if p.requires_grad]
opt = optim.Adam(params_to_train, lr=0.1)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=True)
print([p.shape for p in params_to_train])

criterion = nn.CrossEntropyLoss()

# Train and evaluate
model, hist = train_model(model, dataloaders, criterion,
                          lr_scheduler, num_epochs, device)
