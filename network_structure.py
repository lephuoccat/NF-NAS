#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:56:18 2020

@author: catle
"""
import numpy as np
from scipy.optimize import fsolve

import torch
from torch import nn
from torch.autograd import Variable

if (torch.cuda.is_available() == False):
    device = 'cuda'
else:
    device = 'cpu'
    
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
#        if isinstance(x, tuple):
#            z, sum_log_abs_det_jacobians = x
#        else:
#            z, sum_log_abs_det_jacobians = x, 0
        z = x
        
        # normalize
#        wtu = (self.w @ self.u.t()).squeeze()
#        m_wtu = - 1 + torch.log1p(wtu.exp())
#        u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
       
        # compute transform
        u_hat = self.u
        arg = z @ self.w.t() + self.b
        f_z = z + u_hat*torch.tanh(arg)

        # update log prob.      
        # psi = self.w * (1-torch.tanh(arg)**2)
        # sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + (1 + psi @ u_hat.t()).abs().squeeze().log()
             
        return f_z


# Create NF neural nework with multiple layers
class NF(nn.Module):
    def __init__(self, args):
        super(NF, self).__init__()
        
        # NF
        self.flow = nn.Sequential(*[PlanarFlow(d=args.window_size) for _ in range(args.num_flow)])
       
    def forward(self, X):
        Y = self.flow(X)       
        return Y
   
    def reconstruct(self, Y_tensor, args):
        for i in range(args.num_flow - 1, -1, -1):
            # parameter from NF
            U_tensor = self.flow[i].u
            Y = Y_tensor.cpu().detach().numpy()
            u = self.flow[i].u.cpu().detach().numpy()
            w = self.flow[i].w.cpu().detach().numpy()
            b = self.flow[i].b.cpu().detach().numpy()
            
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
def fit(model, train_loader, alpha1, alpha2, error_list, args):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.MSELoss()
    model.train()
    
    ave_error = 0
    total_error = 0
    
    for epoch in range(args.num_epoch):
        previous_y1 = 0
        previous_y2 = 0
        for batch_idx, (inputs) in enumerate(train_loader):  
            # Pass data (x-domain) into model
            inputs = inputs.to(device)
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
            predict_x = model.reconstruct(reconstruct_target, args)
            # MSE loss from prediction in x-domain
            # only consider the loss of the last value 
            # (the "future" x, but not the "past" x)
            # x_loss = error(predict_x[0, -1], inputs[-1])
            x_loss = error(predict_x, inputs.unsqueeze(0))
            predict_x_loss = error(predict_x[0, -1], inputs[-1])
            
            # Target for y-domain
            target = np.array([[previous_y1, previous_y2, predict_y3.cpu().detach().numpy()]])
            target = Variable(torch.from_numpy(target).float()).to(device)
            # MSE loss from prediction in y-domain
            y_loss = error(output, target)
            
            # loss and backpropagation
            beta = 0.5
            loss = beta * y_loss + (1-beta) * x_loss     
            loss.backward()
            optimizer.step()
            
            # update previous y2 & y3
            previous_y1 = y2.cpu().detach().numpy()
            previous_y2 = y3.cpu().detach().numpy()
            
            # Total error for prediction in x-domain
            total_error += predict_x_loss.cpu().detach().numpy()
            
    print('prediction of last x:')
    print(predict_x.cpu().detach().numpy())
    print('actual x:')
    print(inputs.cpu().detach().numpy())

    ave_error = total_error/(args.num_epoch * (batch_idx+1))
    error_list.append(ave_error)
    # print('The last-batch training MSE: {:.9f}'.format(float(loss.cpu().detach().numpy())))
    print('MSE train: {:.9f}'.format(ave_error))
