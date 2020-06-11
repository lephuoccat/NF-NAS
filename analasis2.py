# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:47:18 2020

@author: Cat Le
"""

import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
# from scipy.optimize import fsolve
from statsmodels.tsa.stattools import levinson_durbin as levinson
from statsmodels.tsa.stattools import acf, ccf
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
parser.add_argument('--batch-size-train', default=3, type=int, help='batch size train')
parser.add_argument('--window-size', default=3, type=int, help='sliding window size for time series data')
parser.add_argument('--num-iteration', default=20, type=int, help='iteration to jointly train NAS')
parser.add_argument('--num-epoch', default=2, type=int, help='number of epochs to train NF')
parser.add_argument('--num-flow', default=10, type=int, help='number of layers in NF')

args = parser.parse_args()

if (torch.cuda.is_available() == False):
    device = 'cuda'
else:
    device = 'cpu'

# ----------------------------------------------
# Load data
filename = "daily-min-temperatures.csv"
# filename = "monthly-sunspots.csv"
# filename = "shampoo.csv"
# filename = "daily-total-female-births.csv"
rows = LoadData(filename)

# Generate sliding data as a trunk
data = np.asarray(rows)
temp = data[:,1]
# temp = temp.astype(float)

# divide data into train and test sets
divider = np.floor(len(temp) * 0.9).astype(int)
temp_train = temp[:divider].astype(float)
temp_test = temp[divider:].astype(float)

# average data
ave_train = []
for i in range(int(temp_train.shape[0]/3)):
    ave_train = np.append(ave_train, np.mean(temp_train[3*i:3*(i+1)]))
    
ave_test = []
for i in range(int(temp_test.shape[0]/3)):
    ave_test = np.append(ave_test, np.mean(temp_test[3*i:3*(i+1)]))

sliding_data_train = window(ave_train, args.window_size+1)
sliding_data_test = window(ave_test, args.window_size+1)

# create array from sliding trunk of data
train_data = []
for value in sliding_data_train:  
    train_data = np.append(train_data, value)
train_data = train_data.astype(float)

test_data = []
for value in sliding_data_test:  
    test_data = np.append(test_data, value)
test_data = test_data.astype(float)
  
# dataloader for neural network training
trainloader = DataLoader(train_data, batch_size=args.batch_size_train + 1, shuffle=False)
testloader = DataLoader(test_data, batch_size=args.batch_size_train + 1, shuffle=False)



# ----------------------------------------------
# main code
# initialize the network structure
cnn = NF(args)
cnn = cnn.to(device)
print(cnn)

# initialize alpha as an array of 1
alpha = torch.ones(args.window_size-1)
error_train = []
error_test = []

# train and test
for i in range(args.num_iteration):
    print('Iteration: %d' % i)
    
    # train network structure
    fit(cnn, trainloader, alpha, error_train, args)
    
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
    output_list = [[] for batch_idx,_ in enumerate(testloader)]
    for batch_idx, (data) in enumerate(testloader):
        # pass test data into trained network
        data = data.to(device)
        x = data[-1].cpu().detach().numpy()
        data = data[:-1]
        features = cnn.flow(data)
        output_list[batch_idx] = np.append(output_list[batch_idx], features.detach().numpy())
        
        # Use alpha parameters from Levinson recursion
        # to predict in y-domain
        y_test = torch.sum(torch.mul(alpha, features[0,1:]))
        
        # calculate predicted y
        reconstruct_y = torch.cat((features[0,1:], y_test.unsqueeze(0)), dim=0)
        
        # pull predicted x from y
        reconstruct_x = cnn.reconstruct(reconstruct_y, args)
    
        # MSE test
        x_test = reconstruct_x.cpu().detach().numpy()[0,-1]
        MSE_test += (x_test - x)**2
    
    MSE_test = MSE_test/(batch_idx+1)
    print('MSE test: %f' % MSE_test)
    error_test.append(MSE_test)
    print('\n')




'''
# Analyze the consistant difference of y-output
output_list = [[] for batch_idx,_ in enumerate(testloader)]
ACF_list = [[] for batch_idx,_ in enumerate(testloader)]
for batch_idx, (data) in enumerate(testloader):
    # pass test data into trained network
    data = data.to(device)
    x = data[-1].cpu().detach().numpy()
    data = data[:-1]
    features = cnn.flow(data)
    ACF_val = acf(features.detach().numpy()[0], unbiased=False, fft=True)
    output_list[batch_idx] = np.append(output_list[batch_idx], features.detach().numpy()) 
    ACF_list[batch_idx] = np.append(ACF_list[batch_idx], ACF_val)
    
m = len(output_list)
n = output_list[0].shape[0]
cum_error = 0
cum_var = 0
for i in range(m - n + 1):
    error = 0
    time_seq = []
    for j in range(n):
        error += abs(output_list[i][-1] - output_list[i+j][-(j+1)])
        time_seq.append(output_list[i+j][-(j+1)])
    var = np.var(time_seq)
    cum_var += var
    cum_error += error/(n-1)
cum_error = cum_error/(m - n + 1)
print(cum_error)
print(cum_var/(m - n + 1))






# histogram
ACF = np.transpose(np.asarray(ACF_list))
np.save('lag1.npy', ACF[1])

# var[lag1]=0.079, var[lag2]=0.0455 , var[lag3]=0.065 , var[lag4]=0.034 , var[lag5]=0.023 
_ = plt.hist(ACF[1], bins='auto')
plt.title('Histogram of autocovariance with lag=1, var=0.079')
plt.show()








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






#--------------------------------
# create multiple time series
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]









temp2 = np.delete(temp,0)
# divide data into train and test sets
divider = np.floor(len(temp2) * 0.9).astype(int)
temp_train = temp2[:divider]
temp_test = temp2[divider:]

sliding_data_train = chunks(temp_train, args.window_size)
sliding_data_test = chunks(temp_test, args.window_size)

# create array from sliding trunk of data
train_data = []
for value in sliding_data_train:  
    if len(value) == args.window_size:
        train_data = np.append(train_data, value)

test_data = []
for value in sliding_data_test:  
    if len(value) == args.window_size:
        test_data = np.append(test_data, value)

# dataloader for neural network training
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=False)

test_data = test_data.astype(float)
testloader = DataLoader(test_data, batch_size=args.batch_size_train, shuffle=False)


# new y output
phi2 = []
for batch_idx, (data) in enumerate(trainloader):
    data = data.to(device)
    
    features = cnn.flow(data).cpu().detach().numpy()
    for i in range(len(features[0])):
        phi2.append(features[0,i])

phi1 = phi[1:]
CCF = ccf(phi1, phi2, unbiased=True)

plt.plot(CCF)
plt.title('cross covariance CCF with lag=1')

'''

