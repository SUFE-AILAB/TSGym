import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import random

# remove randomness
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, scheduler, epoch, lradj,learning_rate, printout=True):
    if lradj == 'type1':
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif lradj == 'type3':
        lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif lradj == 'PEMS':
        lr_adjust = {epoch: learning_rate * (0.95 ** (epoch // 1))}
    elif lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif lradj == 'constant':
        lr_adjust = {epoch: learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_mode=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path,'checkpoint'))
        self.val_loss_min = val_loss

def get_device(use_gpu=True):
    if use_gpu:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f'number of gpu: {n_gpu},cuda name: {torch.cuda.get_device_name(0)},GPU is on')
        else:
            print('GPU is off')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

