import os
import dill
import torch
import importlib
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']