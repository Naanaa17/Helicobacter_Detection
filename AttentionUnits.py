#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:20:27 2023

@author: Guillermo Torres
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


#### Simple Neural Network          
class NeuralNetwork(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        # Get network parameters
        in_dimension = net_params['in_features']
        out_dimension = net_params['out_features']
       
        self.net=nn.Sequential(nn.Linear(in_dimension, round(in_dimension / 2)),  
                            nn.ReLU(inplace=True),
                            nn.Dropout(inplace=False, p=0.5),
                            nn.Linear(round(in_dimension / 2), out_dimension))


    def forward(self, x):
        return self.net(x)
    
#### Attention Neural Netwrok
# Attention Units
class Attention(nn.Module):
    def __init__(self,net_params):
        super(Attention, self).__init__()
        self.M = net_params['in_features'] #Input dimension of the Values NV vectors 
        self.L = net_params['decom_space'] # Dimension of Q(uery),K(eys) decomposition space
        self.ATTENTION_BRANCHES = net_params['ATTENTION_BRANCHES']


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )


    def forward(self, x):

        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        # Attention weights
        A = self.attention(H)  # NVxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV
        
        # Context Vector (Attention Aggregation)
        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM 
        
        return Z, A


class GatedAttention(nn.Module):
    def __init__(self,net_params):
        super(GatedAttention, self).__init__()
        self.M = net_params['in_features'] #Input dimension of the Values NV vectors 
        self.L = net_params['decom_space'] # Dimension of Q(uery),K(eys) decomposition space
        self.ATTENTION_BRANCHES = net_params['ATTENTION_BRANCHES']
        
        # Matrix for Query decomposition
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )
        # Matrix for Keys decomposition
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)


    def forward(self, x):
        # H feature vector matrix  # NV vectors x M dimensions
        H = x.squeeze(0)
        ## Self Attention weights
        # Input Vector Query Decomposition, Q
        A_V = self.attention_V(H)  # NVxL (Projecion of the V input vectors into L dim space)
        # Input Vector Keys Decomposition, K
        A_U = self.attention_U(H)  # NVxL
        # Attention Matrix from Product Q*K 
        A = self.attention_w(A_V * A_U) # element wise multiplication # NVxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxNV
        A = F.softmax(A, dim=1)  # softmax over NV dimension
        
        ## Context Vector (Attention Aggregation)
        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        return Z, A
    

     
