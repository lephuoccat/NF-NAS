#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:56:18 2020

@author: catle
"""
import numpy as np
from scipy.optimize import fsolve

import torch
import torchvision
from torch import nn
from torch.autograd import Variable


# NF network structure based on PlannarFlow
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


# Create NF neural nework with multiple layers
class NF(nn.Module):
    def __init__(self, latent_size, num_layer):
        super(NF, self).__init__()
        
        # NF
        self.flow = nn.Sequential(*[PlanarFlow(d=latent_size) for _ in range(num_layer)])
       
    def forward(self, X):
        Y = self.flow(X)       
        return Y
   
    def reconstruct(self, Y_tensor, num_layer):
        for i in range(num_layer-1, -1, -1):
            # parameter from NF
            U_tensor = self.flow[i].u
            Y = Y_tensor.detach().numpy()
            u = self.flow[i].u.detach().numpy()
            w = self.flow[i].w.detach().numpy()
            b = self.flow[i].b.detach().numpy()
            
            # define equation
            # y = x + theta * tanh(x)
            y = np.dot(w[0], np.transpose(Y)) + b
            theta = np.dot(w[0], np.transpose(u[0]))
            
            # numerical solver
            def f(x):
                return x + theta * np.tanh(x) - y
            
            x = fsolve(f, 3)
            h_x = np.tanh(x)

            # pull the input of each layer
            Y_tensor = Y_tensor - h_x[0] * U_tensor
            
        return Y_tensor
    
# train network structure
def fit(model, train_loader, alpha1, alpha2, EPOCHS, num_layer, error_list):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.MSELoss()
    model.train()
    
    ave_error = 0
    total_error = 0
    
    for epoch in range(EPOCHS):
        previous_y1 = 0
        previous_y2 = 0
        for batch_idx, (inputs) in enumerate(train_loader):  
            # Pass data (x-domain) into model
            optimizer.zero_grad()
            output = model(inputs)
            
            # Output of the NF neural network (y-domain)
            y1 = output[0, 0]
            y2 = output[0, 1]
            y3 = output[0, 2]
            
            # Use alpha parameters from Levinson recursion
            # to predict in y-domain
            predict_y3 = alpha1 * y1 + alpha2 * y2
            
            # Pull x from y-domain
            reconstruct_target = torch.cat((y1.unsqueeze(0), y2.unsqueeze(0), predict_y3.unsqueeze(0)), dim=0)
            predict_x = model.reconstruct(reconstruct_target, num_layer)
            # MSE loss from prediction in x-domain
            x_loss = error(predict_x[0, 2], inputs[2])
            
            
            # Target for y-domain
            target = np.array([[previous_y1, previous_y2, predict_y3.detach().numpy()]])
            target = Variable(torch.from_numpy(target).float())
            # MSE loss from prediction in y-domain
            y_loss = error(output, target)
            
            # loss and backpropagation
            beta = 0.3
            loss = beta * y_loss + (1-beta) * x_loss     
            loss.backward()
            optimizer.step()
            
            # update previous y2 & y3
            previous_y1 = y2.detach().numpy()
            previous_y2 = y3.detach().numpy()
            
            # Total error for prediction in x-domain
            total_error += x_loss.detach().numpy()
            
    print('predict x:')
    print(predict_x.detach().numpy())
    print('actual x:')
    print(inputs.detach().numpy())
    print('The last y1: %f' % y1)
    print('The last y2: %f' % y2)
    print('The last y3: %f' % y3)
    ave_error = total_error/(EPOCHS * (batch_idx+3))
    error_list.append(ave_error)
    print('The last-batch MSE loss: {:.9f}'.format(float(loss.detach().numpy())))
    print('Average MSE loss: {:.9f}'.format(ave_error))
