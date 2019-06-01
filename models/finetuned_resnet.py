import torch.nn as nn
import torch.optim as optim

from util import train_model


class FinetunedResnet:
    def __init__(self, resnet_class, freeze_resnet, num_classes, device):
        model = resnet_class(pretrained=True)

        if freeze_resnet:
            for param in model.parameters():
                param.requires_grad = False

        last_layer_in_size = model.fc.in_features
        model.fc = nn.Linear(last_layer_in_size, num_classes)

        model.to(device)

        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.resnet_class = resnet_class

    def train(self, dataloaders, lr, max_epochs, print_log_file, loss_log_file):
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        opt = optim.Adam(params_to_train, lr=lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=16, verbose=True) #TODO: parametrize, try cos
        print([p.shape for p in params_to_train])

        criterion = nn.CrossEntropyLoss()

        model, hist = train_model(self.model, dataloaders, criterion,
                                  lr_scheduler, max_epochs, self.device, self.num_classes,
                                  print_log_file, loss_log_file)

        return model

