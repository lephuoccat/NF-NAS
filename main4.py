# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:27:29 2020

@author: Cat Le
"""
import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
# from scipy.optimize import fsolve
from statsmodels.tsa.stattools import levinson_durbin as levinson
from load_data import window, LoadData
from IAF import IAFLayer

import torch
# import torchvision
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader

# Parser
parser = argparse.ArgumentParser(description='GWR CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=6, type=int, help='batch size train')
parser.add_argument('--window-size', default=6, type=int, help='sliding window size for time series data')
parser.add_argument('--num-iteration', default=20, type=int, help='iteration to jointly train NAS')
parser.add_argument('--num-epoch', default=2, type=int, help='number of epochs to train NF')
parser.add_argument('--num-flow', default=20, type=int, help='number of layers in NF')

args = parser.parse_args()

if (torch.cuda.is_available() == False):
    device = 'cuda'
else:
    device = 'cpu'

# ----------------------------------------------
# Load data
filename = "daily-min-temperatures.csv"
rows = LoadData(filename)

# Generate sliding data as a trunk
data = np.asarray(rows)
temp = data[:,1]
# divide data into train and test sets
divider = np.floor(len(temp) * 0.9).astype(int)
temp_train = temp[:divider]
temp_test = temp[divider:]

sliding_data_train = window(temp_train, args.window_size + 2)
sliding_data_test = window(temp_test, args.window_size + 2)

# create array from sliding trunk of data
train_data = []
for value in sliding_data_train:  
    train_data = np.append(train_data, value)

test_data = []
for value in sliding_data_test:  
    test_data = np.append(test_data, value)

# dataloader for neural network training
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train + 2, shuffle=False)

test_data = test_data.astype(float)
testloader = DataLoader(test_data, batch_size=args.batch_size_train + 2, shuffle=False)


# ----------------------------------------------
# main code
# initialize the network structure
cnn = NF(args)
cnn = cnn.to(device)
print(cnn)