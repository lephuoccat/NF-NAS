#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:47:23 2020

@author: catle
"""
# importing library
import csv
import os
import argparse
import matplotlib.pyplot as plt

import numpy as np
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
             
        return f_z, sum_log_abs_det_jacobians
   

# VAE class
class VAE_NF(nn.Module):
    def __init__(self, feature_size, latent_size, evolve=True):
        super(VAE_NF, self).__init__()

        # encoder
        self.enc = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(True),
                                 nn.Linear(512, 256), nn.ReLU(True))
        self.enc1 = nn.Linear(256, latent_size)
        self.enc2 = nn.Linear(256, latent_size)

        # decoder
        self.dec = nn.Sequential(nn.Linear(latent_size, 256), nn.ReLU(True),
                                 nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, feature_size))
       
        # NF
        self.evolve = evolve
        self.flow = nn.Sequential(*[PlanarFlow(d=latent_size) for _ in range(32)])
       
        # learnable parameter
        self.gamma = nn.Parameter(torch.randn(1))
       
    def encode(self, x):
        h1 = self.enc(x)
        mu_z = self.enc1(h1)
        logvar_z = self.enc2(h1)
       
        return mu_z, logvar_z
   
    def decode(self, z):
        h1 = self.dec(z)
        x_hat = torch.sigmoid(h1)

        return x_hat
   
    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        std_z = (0.5*logvar_z).exp()
        z = mu_z + std_z * torch.randn_like(std_z)
       
        # pass z through the NF
        if self.evolve is True:
            z_k, sum_log_abs_det_jacobians = self.flow(z)
            # KL = -0.5*((z-mu_z)/std_z).pow(2).sum() - 0.5*logvar_z.sum() + 0.5*z.pow(2).sum() - sum_log_abs_det_jacobians.sum()
            KL = -0.5*torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) - sum_log_abs_det_jacobians.sum()
        else:
            # KL = -0.5*((z-mu_z)/std_z).pow(2).sum() - 0.5*logvar_z.sum() + 0.5*z.pow(2).sum()
            KL = -0.5*torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            z_k = z
       
        # decode
        x_hat = self.decode(z_k)
       
        return x_hat, KL
   
    def vae_loss(self, x, x_hat, KL):
        # BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        # MSE = 0.5*torch.log(self.gamma**2)*x.shape[0]*x.shape[0] + (x-x_hat).pow(2).sum() / 2*self.gamma**2
        MSE = 0.5*(x-x_hat).pow(2).sum()
        loss = KL + MSE

        return loss