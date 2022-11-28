# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:21:11 2021

@author: marnold
"""

import torch
import torch.nn as nn
import sys
from libraries.general import *
import numpy as np

class ae_camera_model(nn.Module):
    def __init__(self,args,in_dimensions):
        super(ae_camera_model, self).__init__()
        self.in_dim0 = in_dimensions[2]
        self.in_dim1 = in_dimensions[3]
        self.encoder = Encoder(args, args["feat_dim"], in_dimensions[2], in_dimensions[3])
        self.decoder = Decoder(args, args["feat_dim"], self.encoder.conv_dim, in_dimensions[2], in_dimensions[3],self.encoder.shape_arr)

    def forward(self,x):
        # Go through encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x,self.encoder.ids_arr)
        return x

    def defineLoss(self,loss):
        self.calc_loss = defined_losses(loss)

    def trainNN(self, input_data, target):
        # Pass Through NN
        output = self.forward(input_data)

        # Get Loss
        loss = self.calc_loss(output,target)

        # Backpropagate
        loss.backward()

        return loss.item()

    def testNN(self,input_data,target):
        # Pass Through NN
        output = self.forward(input_data)

        # Get Loss
        loss = self.calc_loss(output,target)

        return loss.item()


class Encoder(nn.Module):
    def __init__(self, args,feature_dim,radio_dim_0,radio_dim_1):
        super().__init__()
        self.radio_dim_0 = radio_dim_0
        self.radio_dim_1 = radio_dim_1
        # Parse Args Encoder
        # Conv sizes
        c1_size = args['c1_size']
        c2_size = args['c2_size']
        c3_size = args['c3_size']
        c4_size = args['c4_size']
        c5_size = args['c5_size']
        c6_size = args['c6_size']
        c7_size = args['c7_size']
        c8_size = args['c8_size']
        c9_size = args['c9_size']
        c10_size = args['c10_size']
        # kernel sizes
        k1_size = args['k1_size']
        k2_size = args['k2_size']
        k3_size = args['k3_size']
        k4_size = args['k4_size']
        k5_size = args['k5_size']
        k6_size = args['k6_size']
        k7_size = args['k7_size']
        k8_size = args['k8_size']
        k9_size = args['k9_size']
        k10_size = args['k10_size']
        # max sizes
        m1_size = args['m1_size']
        m2_size = args['m2_size']
        m3_size = args['m3_size']
        m4_size = args['m4_size']
        m5_size = args['m5_size']
        m6_size = args['m6_size']
        m7_size = args['m7_size']
        m8_size = args['m8_size']
        m9_size = args['m9_size']
        m10_size = args['m10_size']
        # stride sizes
        s1_size = args['s1_size']
        s2_size = args['s2_size']
        s3_size = args['s3_size']
        s4_size = args['s4_size']
        s5_size = args['s5_size']
        s6_size = args['s6_size']
        s7_size = args['s7_size']
        s8_size = args['s8_size']
        s9_size = args['s9_size']
        s10_size = args['s10_size']

        # Get activation function
        act_func = map_act_func(args['act_func'])
        last_act_func = map_act_func(args['last_act_func'])

        ### Convolutional section
        self.encoder_cnn_layer_1 = nn.Sequential(
            nn.Conv2d(3, c1_size, kernel_size=(k1_size,k2_size), stride=(s1_size,s2_size),padding=1),
            act_func,
        )
        self.encoder_cnn_layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(m1_size,m2_size), stride=(s1_size,s2_size), return_indices=True),
        )
        self.encoder_cnn_layer_3 = nn.Sequential(
             nn.Conv2d(c1_size, c2_size, kernel_size=(k3_size,k4_size), stride=(s3_size,s4_size),padding=1),
             act_func,
         )
        self.encoder_cnn_layer_4 = nn.Sequential(
             nn.MaxPool2d(kernel_size=(m3_size,m4_size), stride=(s3_size,s4_size), return_indices=True),
        )
        self.encoder_cnn_layer_5 = nn.Sequential(
             nn.Conv2d(c2_size, c3_size, kernel_size=(k5_size,k6_size), stride=(s5_size,s6_size),padding=1),
             act_func,
        )
        self.encoder_cnn_layer_6 = nn.Sequential(
             nn.MaxPool2d(kernel_size=(m5_size,m6_size), stride=(s5_size,s6_size), return_indices=True),
        )

        # Get Shape of output Neck
        self.conv_dim, self.shape_arr = self._get_output_shape()

        # Linear Section
        self.encoder_lin = nn.Sequential(
            nn.Linear(len(self.conv_dim.flatten()),feature_dim),
            last_act_func,
        )

    # Generate fake input sample, such it can pass through till the linear part
    def _get_output_shape(self):
        shape_arr =[]
        t = torch.zeros(1,3,self.radio_dim_0,self.radio_dim_1)#settings.rad_dim[0], settings.rad_dim[1])
        shape_arr.append(t.shape)
        t = self.encoder_cnn_layer_1(t)
        shape_arr.append(t.shape)
        t, ids = self.encoder_cnn_layer_2(t)
        shape_arr.append(t.shape)
        t = self.encoder_cnn_layer_3(t)
        shape_arr.append(t.shape)
        t, ids = self.encoder_cnn_layer_4(t)
        shape_arr.append(t.shape)
        t = self.encoder_cnn_layer_5(t)
        shape_arr.append(t.shape)
        t, ids = self.encoder_cnn_layer_6(t)
        shape_arr.append(t.shape)

        #t = self.encoder_cnn(t)
        return t, shape_arr


    def forward(self, x):
        ids_arr = []
        shape_arr =[]
        x = self.encoder_cnn_layer_1(x)
        x, ids = self.encoder_cnn_layer_2(x)
        ids_arr.append(ids)
        x = self.encoder_cnn_layer_3(x)
        x, ids = self.encoder_cnn_layer_4(x)
        ids_arr.append(ids)
        x = self.encoder_cnn_layer_5(x)
        x, ids = self.encoder_cnn_layer_6(x)
        ids_arr.append(ids)
        #x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_lin(x)
        self.ids_arr = ids_arr
        return x

class Decoder(nn.Module):
    def __init__(self, args,feature_dim,conv_dim,radio_dim_0,radio_dim_1,shape_arr):
        super().__init__()
        self.radio_dim_0 = radio_dim_0
        self.radio_dim_1 = radio_dim_1
        self.shape_arr = shape_arr
        self.conv_dim = conv_dim
        # Parse Args Decoder
        # Conv sizes
        c1_size = args['c1_size']
        c2_size = args['c2_size']
        c3_size = args['c3_size']
        c4_size = args['c4_size']
        c5_size = args['c5_size']
        c6_size = args['c6_size']
        c7_size = args['c7_size']
        c8_size = args['c8_size']
        c9_size = args['c9_size']
        c10_size = args['c10_size']
        # kernel sizes
        k1_size = args['k1_size']
        k2_size = args['k2_size']
        k3_size = args['k3_size']
        k4_size = args['k4_size']
        k5_size = args['k5_size']
        k6_size = args['k6_size']
        k7_size = args['k7_size']
        k8_size = args['k8_size']
        k9_size = args['k9_size']
        k10_size = args['k10_size']
        # max sizes
        m1_size = args['m1_size']
        m2_size = args['m2_size']
        m3_size = args['m3_size']
        m4_size = args['m4_size']
        m5_size = args['m5_size']
        m6_size = args['m6_size']
        m7_size = args['m7_size']
        m8_size = args['m8_size']
        m9_size = args['m9_size']
        m10_size = args['m10_size']
        # stride sizes
        s1_size = args['s1_size']
        s2_size = args['s2_size']
        s3_size = args['s3_size']
        s4_size = args['s4_size']
        s5_size = args['s5_size']
        s6_size = args['s6_size']
        s7_size = args['s7_size']
        s8_size = args['s8_size']
        s9_size = args['s9_size']
        s10_size = args['s10_size']

        # Get activation function
        act_func = map_act_func(args['act_func'])
        last_act_func = map_act_func(args['last_act_func'])

        # Linear decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(feature_dim,len(conv_dim.flatten())),
        )
        # Unflatten
        self.unflatten = nn.Unflatten(dim=1,
        unflattened_size=conv_dim.shape)

        ## Correct for rounding of layer outputs right
        step_1 =  (256+2*1-k2_size)/s2_size +1
        step_2 = (np.floor(step_1)+2*0-m2_size)/s2_size +1
        step_3 = (np.floor(step_2)+2*1-k4_size)/s4_size +1
        step_4 = (np.floor(step_3)+2*0-m4_size)/s4_size +1
        step_5 = (np.floor(step_4)+2*1-k6_size)/s6_size +1
        step_6 = (np.floor(step_5)+2*0-m6_size)/s6_size +1
        if (step_6 % 1 !=0):
             m6_size = m6_size + 1  #+ int(np.floor(s6_size/2))
        if (step_5 % 1 !=0):
             k6_size = k6_size + 1#+ int(np.floor(s6_size/2))#+inc; inc+=1
        if (step_4 % 1 !=0):
             m4_size = m4_size + 1#+ int(np.floor(s4_size/ 2))#+inc; inc+=1
        if (step_3 % 1 !=0):
             k4_size = k4_size + 1# + int(np.floor(s4_size/2))#+inc; inc+=1
        if (step_2 % 1 !=0):
             m2_size = m2_size + 1 #+ int(np.floor(s2_size/2))#+inc; inc+=1
        if (step_1 % 1 !=0):
             k2_size = k2_size + 1 # + int(np.floor(s2_size/2))#+inc; inc+=1

        ## Correct for rounding of layer outputs left
        step_1 =  (256+2*1-k1_size)/s1_size +1
        step_2 = (np.floor(step_1)+2*0-m1_size)/s1_size +1
        step_3 = (np.floor(step_2)+2*1-k3_size)/s3_size +1
        step_4 = (np.floor(step_3)+2*0-m3_size)/s3_size +1
        step_5 = (np.floor(step_4)+2*1-k5_size)/s5_size +1
        step_6 = (np.floor(step_5)+2*0-m5_size)/s5_size +1
        if (step_6 % 1 !=0):
             m5_size = m5_size + 1  #+ int(np.floor(s6_size/2))
        if (step_5 % 1 !=0):
             k5_size = k5_size + 1#+ int(np.floor(s6_size/2))#+inc; inc+=1
        if (step_4 % 1 !=0):
             m3_size = m3_size + 1#+ int(np.floor(s4_size/ 2))#+inc; inc+=1
        if (step_3 % 1 !=0):
             k3_size = k3_size + 1# + int(np.floor(s4_size/2))#+inc; inc+=1
        if (step_2 % 1 !=0):
             m1_size = m1_size + 1 #+ int(np.floor(s2_size/2))#+inc; inc+=1
        if (step_1 % 1 !=0):
             k1_size = k1_size + 1 # + int(np.floor(s2_size/2))#+inc; inc+=1



        ### Convolutional section
        self.unpool_1 = nn.MaxUnpool2d(kernel_size=(m5_size,m6_size), stride=(s5_size,s6_size))
        self.decoder_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(c3_size, c2_size, kernel_size=(k5_size,k6_size), stride=(s5_size,s6_size),padding=1),
            act_func,
        )
        self.unpool_2 = nn.MaxUnpool2d(kernel_size=(m3_size,m4_size), stride=(s3_size,s4_size))
        self.decoder_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(c2_size, c1_size, kernel_size=(k3_size,k4_size), stride=(s3_size,s4_size),padding=1),
            act_func,
        )
        self.unpool_3 = nn.MaxUnpool2d(kernel_size=(m1_size,m2_size), stride=(s1_size,s2_size))
        self.decoder_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(c1_size,3, kernel_size=(k1_size,k2_size), stride=(s1_size,s2_size),padding=1),
            last_act_func,
        )





    def forward(self, x,ids_arr):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        #print(x.shape)
        x = self.unpool_1(x[:,0,...],ids_arr[-1])
        #print(x.shape)
        x = self.decoder_layer_1(x)
        #print(x.shape)
        x = self.unpool_2(x,ids_arr[-2])
        #print(x.shape)
        x = self.decoder_layer_2(x)
        #print(x.shape)
        x = self.unpool_3(x,ids_arr[-3])
        #print(x.shape)
        x = self.decoder_layer_3(x)
        #print(x.shape)

        return x
