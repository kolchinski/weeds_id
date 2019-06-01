
import copy
import time

import numpy as np
import torch


# Code adapted from PyTorch fine tuning tutorial at
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Epoch offset is for when we have multiple phases, to keep track of the overall epoch
def train_model(model, dataloaders, criterion, lr_scheduler, num_epochs, device, num_classes,
                save_every, log_dir, print_log_file, loss_log_file,
                epoch_offset=0):

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=print_log_file)
        print('-' * 10, file=print_log_file)

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
                lr_scheduler.optimizer.zero_grad()

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), file=print_log_file)

            for i in range(num_classes):
                print('Class {} ({}): Accuracy {}/{} ({}%)'.format(
                    i, dataloaders[phase].dataset.dataset.classes[i], confusion_matrix[i,i],
                    confusion_matrix[i].sum(), 100.*confusion_matrix[i,i]/confusion_matrix[i].sum()
                ), file=print_log_file)
            print(confusion_matrix, file=print_log_file)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                #lr_scheduler.step(epoch_loss) for reduce-on-plateau LR Scheduler
                lr_scheduler.step()
                loss_log_file.write('{},{}\n'.format(loss, epoch_acc))
            else:
                loss_log_file.write('{},{},{},'.format(epoch + epoch_offset, loss, epoch_acc))

        print(file=print_log_file)
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), log_dir + '/model_dict_{}.pth'.format(epoch + epoch_offset))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=print_log_file)
    print('Best val Acc: {:4f}'.format(best_acc), file=print_log_file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
