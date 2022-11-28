# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:21:11 2021
lidar_only position NN
@author: marnold
"""

import torch
import torch.nn as nn
import sys
from libraries.general import *
current_loss =  nn.CrossEntropyLoss()
# Dense Model
class dense_model(nn.Module):
    def __init__(self,args,in_dim,out_dim):
        super(dense_model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Linear size
        lin1_size = args['lin1_size']
        lin2_size = args['lin2_size']
        lin3_size = args['lin3_size']
        lin4_size = args['lin4_size']

        # Get activation function
        act_func = map_act_func(args['act_func'])
        last_act_func = map_act_func(args['last_act_func'])

        # Get Shape of output Neck
        self.linear_layer = nn.Sequential(
            nn.Linear(self.in_dim,lin1_size),
            act_func,
            nn.Linear(lin1_size,lin2_size),
            act_func,
            nn.Linear(lin2_size,lin3_size),
            act_func,
            nn.Linear(lin3_size,lin4_size),
            act_func,
            nn.Linear(lin4_size,self.out_dim),
            last_act_func,
        )


    def forward(self,x):
        # Flatten input
        x = x.view(x.size(0), -1)

        # Go through linear layers
        x = self.linear_layer(x)

        # Return output
        return x

    def defineLoss(self,loss):
        self.calc_loss = defined_losses(loss)
        if (loss =="CE"):
           self.class_loss =1
        else:
           self.class_loss =0

    def trainNN(self, input_data, target):
        # Pass Through NN
        output = self.forward(input_data)
        # Change target if class loss
        if(self.class_loss):
           target = torch.max(target, 1)[1]

        # Get Loss
        loss = self.calc_loss(output,target)

        # Backpropagate
        loss.backward()

        return loss.item()

    def testNN(self,input_data,target):
        # Pass Through NN
        output = self.forward(input_data)

        # Get Loss
        if(self.class_loss):
            target = torch.max(target, 1)[1]
            l = nn.CrossEntropyLoss()
            loss =  l(output,target)
        else:
           loss =  self.calc_loss(output,target)

        return loss.item()


