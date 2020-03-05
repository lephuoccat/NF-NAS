# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:40:15 2020

@author: Cat Le
"""


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
# from scipy.optimize import fsolve
from statsmodels.tsa.stattools import levinson_durbin as levinson
from load_data import window, LoadData
from network_structure2 import NF, fit

import torch
# import torchvision
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader

# Parser
parser = argparse.ArgumentParser(description='GWR CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=5, type=int, help='batch size train')
parser.add_argument('--window-size', default=5, type=int, help='sliding window size for time series data')
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

sliding_data_train = window(temp_train, args.window_size + 1)
sliding_data_test = window(temp_test, args.window_size + 1)

# create array from sliding trunk of data
train_data = []
for value in sliding_data_train:  
    train_data = np.append(train_data, value)

test_data = []
for value in sliding_data_test:  
    test_data = np.append(test_data, value)

# dataloader for neural network training
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train + 1, shuffle=False)

test_data = test_data.astype(float)
testloader = DataLoader(test_data, batch_size=args.batch_size_train + 1, shuffle=False)

# ----------------------------------------------
# main code
# initialize the network structure
cnn = NF(args.window_size, args.num_flow)
cnn = cnn.to(device)
print(cnn)

# initialize alpha as an array of 1
alpha = torch.ones(args.window_size - 1)
error_train = []
error_test = []

# train and test
for i in range(args.num_iteration):
    print('Iteration: %d' % i)
    
    # train network structure
    fit(cnn, trainloader, alpha, args.num_epoch, args.num_flow, error_train)
    
    # extract latent features from trained network
    phi = []
    for batch_idx, (data) in enumerate(trainloader):
        data = data[:-1]
        data = data.to(device)
        
        features = cnn.flow(data).cpu().detach().numpy()
        
        # add the first 2 features in the 1st trunk
        if batch_idx == 0:
            for i in range(len(features[0]) - 1):
                phi.append(features[0,i])
        # add the last features (predicting y) in every trunk
        phi.append(features[0,-1])
    
    # train alpha parameter with RLS
    [_,alpha,_,_,_] = levinson(phi, nlags=args.window_size-1)
    
    # convert alpha to tensor
    alpha = torch.from_numpy(alpha).type(torch.FloatTensor)
    
    
    # test
    MSE_test = 0
    for batch_idx, (data) in enumerate(testloader):
        # pass test data into trained network
        data = data.to(device)
        x = data[-1].cpu().detach().numpy()
        data = data[:-1]
        features = cnn.flow(data)
            
        # Use alpha parameters from Levinson recursion
        # to predict in y-domain
        y_test = torch.sum(torch.mul(alpha, features[0,1:]))
        
        # calculate predicted y
        reconstruct_y = torch.cat((features[0,1:], y_test.unsqueeze(0)), dim=0)
        
        # pull predicted x from y
        reconstruct_x = cnn.reconstruct(reconstruct_y, args.num_flow)
    
        # MSE test
        x_test = reconstruct_x.cpu().detach().numpy()[0,-1]
        MSE_test += (x_test - x)**2
    
    MSE_test = MSE_test/(batch_idx+1)
    print('MSE test: %f' % MSE_test)
    error_test.append(MSE_test)
    print('\n')







#------------------------------------
# plot the MSE loss
t = np.arange(len(error_train))

# loss
plt.figure(figsize=[14,10])
plt.plot(t, error_train, 'r', label="train")
plt.plot(t, error_test, 'b', label="test")
plt.title('MSE Loss')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend(['train',
            'test'])
#plt.axis([0, 19, -5, 75])
plt.xticks(np.arange(0, 20, step=1))
plt.grid(True)
plt.show()
