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
from network_structure import NF, fit

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
    
# ----------------------------------------------
# Load data
filename = "daily-min-temperatures.csv"
rows = LoadData(filename)

# Generate sliding data as a trunk
data = np.asarray(rows)
temp = data[:,1]
sliding_data = window(temp, args.window_size)

# create array from sliding trunk of data
train_data = []
for value in sliding_data:  
    train_data = np.append(train_data, value)

# dataloader for neural network training
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=False)


# ----------------------------------------------
# main code
cnn = NF(args.window_size, args.num_flow)
print(cnn)

alpha1 = 1
alpha2 = 1
error_list = []
alpha1_list = []
alpha2_list = []

for i in range(args.num_iteration):
    print('Iteration: %d' % i)
    print('alpha 1: %f' % alpha1)
    print('alpha 2: %f' % alpha2)
    
    fit(cnn, trainloader, alpha1, alpha2, args.num_epoch, args.num_flow, error_list)
    
    phi = []
    for _, (data) in enumerate(trainloader):
        features = cnn.flow(data).cpu().detach().numpy()
        phi.append(features[0,2])
    
    [a1,alpha,a2,a3,a4] = levinson(phi, nlags=2)
    alpha1 = alpha[0]
    alpha2 = alpha[1]
    alpha1_list.append(alpha1)
    alpha2_list.append(alpha2)
    
    print('\n')









#------------------------------------
# plot the loss and alpha parameters
t = np.arange(len(error_list))

# loss
plt.figure(figsize=[14,10])
plt.plot(t, error_list, 'r')
plt.title('MSE Loss')
plt.xlabel('number of epochs')
plt.ylabel('loss')
#plt.axis([0, 19, -5, 75])
plt.xticks(np.arange(0, 20, step=1))
plt.grid(True)
plt.show()

# alpha parameters
plt.figure(figsize=[14,10])
plt.plot(t, alpha1_list, 'bs-', t, alpha2_list, 'g^-')
plt.title('\N{GREEK SMALL LETTER ALPHA} parameters')
plt.xlabel('number of epochs')
plt.ylabel('\N{GREEK SMALL LETTER ALPHA}')
plt.legend(['\N{GREEK SMALL LETTER ALPHA}\N{SUBSCRIPT ONE}',
            '\N{GREEK SMALL LETTER ALPHA}\N{SUBSCRIPT TWO}'])
plt.xticks(np.arange(0, 20, step=1))
plt.grid(True)
plt.show()












