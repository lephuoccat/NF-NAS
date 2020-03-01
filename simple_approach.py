#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:28:14 2020

@author: catle
"""

# importing library
import csv
import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import fsolve
from numpy import linalg as LA
from itertools import islice
from statsmodels.tsa.stattools import levinson_durbin as levinson

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision.datasets import MNIST
#from torchvision.datasets import FashionMNIST
#from torchvision.utils import save_image

#from NF_PlannarFlow import PlanarFlow

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

# Sliding window
def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

#-----------------------
# load dataset
filename = "daily-min-temperatures.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = []

# load csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
    
    # extracting field names through first row 
    fields = next(csvreader)
    
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
        
    # get total number of rows 
    print("Total number of rows: %d"%(csvreader.line_num)) 

# printing the field names 
print('Field names: ' + ', '.join(field for field in fields)) 
  



# -------------------------------
# Network structure training
# generate sliding data
data = np.asarray(rows)
temp = data[:,1]
sliding_data = window(temp, args.window_size)

train_data = []
for value in sliding_data:  
    train_data = np.append(train_data, value)

# loader for train data
train_data = train_data.astype(float)
trainloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=False)

# create the network structure
class PlanarFlow(nn.Module):
    def __init__(self, d=3, init_sigma=0.01):
        """
        d : latent space dimensnion
        init_sigma : var of the initial parameters
        """
        super(PlanarFlow, self).__init__()
        self.d = d
        self.u = nn.Parameter(torch.randn(1, d).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, d).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))
       
       
    def forward(self, x, normalize_u=True):  
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0
       
        # normalize
#        wtu = (self.w @ self.u.t()).squeeze()
#        m_wtu = - 1 + torch.log1p(wtu.exp())
#        u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
       
        # compute transform
        u_hat = self.u
        arg = z @ self.w.t() + self.b
        f_z = z + u_hat*torch.tanh(arg)

        # update log prob.      
        psi = self.w * (1-torch.tanh(arg)**2)
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + (1 + psi @ u_hat.t()).abs().squeeze().log()
             
        return f_z


# normalizing flow (invertible) neural nework
class NF(nn.Module):
    def __init__(self, latent_size):
        super(NF, self).__init__()

        # NF
        self.flow = nn.Sequential(*[PlanarFlow(d=latent_size) for _ in range(args.num_flow)])
       
    def forward(self, X):
        Y = self.flow(X)       
        return Y
   
    def reconstruct(self, Y_tensor):
        for i in range(args.num_flow-1, -1, -1):
            # parameter from NF
            U_tensor = self.flow[i].u
            Y = Y_tensor.detach().numpy()
            u = self.flow[i].u.detach().numpy()
            w = self.flow[i].w.detach().numpy()
            b = self.flow[i].b.detach().numpy()
            
            # define equation
            y = np.dot(w[0], np.transpose(Y)) + b
            theta = np.dot(w[0], np.transpose(u[0]))
            
            def f(x):
                return x + theta * np.tanh(x) - y
            
            x = fsolve(f, 3)
            h_x = np.tanh(x)

            # pull the input of each layer
            Y_tensor = Y_tensor - h_x[0] * U_tensor
            
        return Y_tensor


    
# train network structure
def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.MSELoss()
    EPOCHS = args.num_epoch
    model.train()
    
    ave_error = 0
    total_error = 0
    
    for epoch in range(EPOCHS):
        previous_y1 = 0
        previous_y2 = 0
        for batch_idx, (inputs) in enumerate(train_loader):  
            # pass data into model
            optimizer.zero_grad()
            output = model(inputs)
            
            # the output of the NF neural network
            y1 = output[0, 0]
            y2 = output[0, 1]
            y3 = output[0, 2]
            
            # Use alpha parameters from Levinson recursion
            predict_y3 = alpha1 * y1 + alpha2 * y2
            
            # reconstruct x from y
            reconstruct_target = torch.cat((y1.unsqueeze(0), y2.unsqueeze(0), predict_y3.unsqueeze(0)), dim=0)
            predict_x = model.reconstruct(reconstruct_target)
#            predict_x = Variable(torch.from_numpy(predict_x).float())
            
            # loss of reconstruction in x domain
            reconstruct_loss = error(predict_x[0, 2], inputs[2])
            
            
            # define the target for loss in y domain
            target = np.array([previous_y1, previous_y2, predict_y3.detach().numpy()])
            target = Variable(torch.from_numpy(target).float())
#            target = torch.cat((previous_y1.unsqueeze(0), previous_y2.unsqueeze(0), predict_y3.unsqueeze(0)), dim=0)
            
            # loss and backpropagation
            beta = 0.3
            loss = beta * error(output, target) + (1-beta) * reconstruct_loss     
            loss.backward()
            optimizer.step()
            
            # update previous y2 & y3
            previous_y1 = y2.detach().numpy()
            previous_y2 = y3.detach().numpy()
            
            # Total error
            total_error += reconstruct_loss.detach().numpy()
            
            # print the last MSE loss
#            if batch_idx == (temp.shape[0]-2-1):
#                print('The last MSE loss: {:.9f}'.format(float(loss.detach().numpy())))
    
    print('predict x:')
    print(predict_x.detach().numpy())
    print('actual x')
    print(inputs.detach().numpy())
    print('The last y1: %f' % y1)
    print('The last y2: %f' % y2)
    print('The last y3: %f' % y3)
    ave_error = total_error/(EPOCHS * temp.shape[0])
    error_list.append(ave_error)
    print('The last-batch MSE loss: {:.9f}'.format(float(loss.detach().numpy())))
    print('Average MSE loss: {:.9f}'.format(ave_error))



#---------------------
# main code
#cnn = Phi().cuda()
#cnn = CNN()
cnn = NF(args.window_size)
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
    
    fit(cnn, trainloader)
    
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


