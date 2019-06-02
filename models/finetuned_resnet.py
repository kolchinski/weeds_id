import types

import torch
import torch.nn as nn
import torch.optim as optim

from util import train_model



def forward_spatial_first(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # Change from BxCxHxW to BxHxWxC, where C is number of channels
    x = x.permute(0,2,3,1)

    #BxHxWxC -> BxHxWxN, where N is number of output classes
    x = self.fc(x)

    # Go back from BxHxWxN to BxNxHxW
    x = x.permute(0, 3, 1, 2)

    # Go from BxNxHxW to BxNx1x1
    x = self.avgpool(x)

    # and to BxN
    x = x.reshape(x.size(0), -1)

    return x


class FinetunedResnet:
    def __init__(self, resnet_class, num_classes, device):
        model = resnet_class(pretrained=True)

        # Freeze all pretrained Resnet layers by default (new layer not frozen)
        for param in model.parameters():
            param.requires_grad = False
        last_layer_in_size = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_size, num_classes)

        # Reorder the pretrained model's last two layers
        model.forward = types.MethodType(forward_spatial_first, model)

        model.to(device)

        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.resnet_class = resnet_class

    def train_from_scratch(self, dataloaders, lr, finetune_lr, max_epochs, save_every,
              log_dir, print_log_file, loss_log_file):

        params_to_train = [p for p in self.model.parameters() if p.requires_grad]

        opt = optim.Adam(params_to_train, lr=lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        criterion = nn.CrossEntropyLoss()

        model, hist = train_model(self.model, dataloaders, criterion, lr_scheduler,
                                  max_epochs, self.device, self.num_classes, save_every,
                                  log_dir, print_log_file, loss_log_file)

        model, hist = self.finetune_whole_model(dataloaders, finetune_lr, max_epochs, save_every,
                                                log_dir, print_log_file, loss_log_file)
        return model, hist

    def finetune_whole_model(self, dataloaders, lr, max_epochs, save_every,
                             log_dir, print_log_file, loss_log_file,
                             pretrained_model_path=None):

        # If path is passed, start training from the model provided
        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path))

        for param in self.model.parameters():
            param.requires_grad = True
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]

        opt = optim.Adam(params_to_train, lr=lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        criterion = nn.CrossEntropyLoss()

        model, hist = train_model(self.model, dataloaders, criterion, lr_scheduler,
                                  max_epochs, self.device, self.num_classes, save_every,
                                  log_dir, print_log_file, loss_log_file,
                                  epoch_offset=max_epochs)

        return model, hist


