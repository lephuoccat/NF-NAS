#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:42:10 2020

@author: catle
"""
import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import fsolve
from statsmodels.tsa.stattools import levinson_durbin as levinson
from load_data import window, LoadData

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Parser
parser = argparse.ArgumentParser(description='GWR CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=3, type=int, help='batch size train')
parser.add_argument('--num-iteration', default=20, type=int, help='iteration to train NAS')
parser.add_argument('--num-epoch', default=1, type=int, help='number of epochs')
parser.add_argument('--num-flow', default=10, type=int, help='number of flows')
parser.add_argument('--window-size', default=3, type=int, help='sliding window size for time series data')
args = parser.parse_args()

if (torch.cuda.is_available() == 'True'):
    device = 'cuda'
else:
    device = 'cpu'
    

# Load data
filename = "daily-min-temperatures.csv"
rows = LoadData(filename)

# Generate sliding data as a trunk
data = np.asarray(rows)
temp = data[:,1]
sliding_data = window(temp, args.window_size)

train_data = []
for value in sliding_data:  
    train_data = np.append(train_data, value)

# dataloader for neural network training
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=False)














