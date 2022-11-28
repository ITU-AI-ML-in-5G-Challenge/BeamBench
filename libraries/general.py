#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:39:38 2022

@author: maximilian
"""
# Import libraries
import torch
import torch.nn as nn
import sys
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
import numpy as np
import pandas as pd
import utm

# Radar Settings
RADAR_PARAMS = {'chirps':            128, # number of chirps per frame
                'tx':                  1, # transmitter antenna elements
                'rx':                  4, # receiver antenna elements
                'samples':           256, # number of samples per chirp
                'adc_sampling':      5e6, # Sampling rate [Hz]
                'chirp_slope': 15.015e12, # Ramp (freq. sweep) slope [Hz/s]
                'start_freq':       77e9, # [Hz]
                'idle_time':           5, # Pause between ramps [us]
                'ramp_end_time':      60} # Ramp duration [us]
samples_per_chirp = RADAR_PARAMS['samples']
n_chirps_per_frame = RADAR_PARAMS['chirps']
C = 3e8
chirp_period = (RADAR_PARAMS['ramp_end_time'] + RADAR_PARAMS['idle_time']) * 1e-6
RANGE_RES = ((C * RADAR_PARAMS['adc_sampling']) /
                    (2*RADAR_PARAMS['samples'] * RADAR_PARAMS['chirp_slope']))
FoV =110

def minmax(arr):
    return (arr - arr.min())/ (arr.max()-arr.min())

def range_velocity_map(data):
    data = np.fft.fft(data, axis=1) # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = data/np.max(data)
    #data = np.log(1+data)
    return data

def range_angle_map(data, fft_size = 64):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis = 0) # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    data = data/np.max(data)
    return data.T


#%% Normalize positions (min-max XY position difference between UE and BS)
def xy_from_latlong(lat_long):
    """
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)

# Taken from https://github.com/DeepSense6G/Multi-Modal-Beam-Prediction-Challenge-2022-Baseline/
def compute_DBA_score(y_pred, y_true, max_k=3, delta=5):
    """
    The top-k MBD (Minimum Beam Distance) as the minimum distance
    of any beam in the top-k set of predicted beams to the ground truth beam.

    Then we take the average across all samples.

    Then we average that number over all the considered Ks.
    """
    n_samples = y_pred.shape[0]
    n_beams = y_pred.shape[-1]

    yk = np.zeros(max_k)
    for k in range(max_k):
        acc_avg_min_beam_dist = 0
        idxs_up_to_k = np.arange(k+1)
        for i in range(n_samples):
            aux1 = np.abs(y_pred[i, idxs_up_to_k] - y_true[i]) / delta
            # Compute min between beam diff and 1
            aux2 = np.min(np.stack((aux1, np.zeros_like(aux1)+1), axis=0), axis=0)
            acc_avg_min_beam_dist += np.min(aux2)

        yk[k] = 1 - acc_avg_min_beam_dist / n_samples

    return np.mean(yk)

# Taken from https://github.com/DeepSense6G/Multi-Modal-Beam-Prediction-Challenge-2022-Baseline/
def save_pred_to_csv(y_pred, top_k=[1,2,3], target_csv='beam_pred.csv'):
    """
    Saves the predicted beam results to a csv file.
    Expects y_pred: n_samples x N_BEAMS, and saves the top_k columns only.
    """

    cols = [f'top-{i} beam' for i in top_k]
    df = pd.DataFrame(data=y_pred[:, np.array(top_k)-1], columns=cols)
    df.index.name = 'index'
    df.to_csv(target_csv)

# Define matching functions
## NEED TO Fix dimensions
def match_radar_seq(radar_labels,gt_pos):
    # Angle lim
    ang_lim = 75 # comes from array dimensions and frequencies

    # Output Array
    radar_est = []

    # Convert radar labels to range az
    label_first_frame = np.array( [(256-radar_labels[0,:,1])/(2*RANGE_RES),ang_lim*(radar_labels[0,:,0]-32)/64]).T
    label_second_frame = np.array( [(256-radar_labels[1,:,1])/(2*RANGE_RES),ang_lim*(radar_labels[1,:,0]-32)/64]).T

    # Match closest angle (AS angle is the most important here anyhow)
    min_1 = np.min(abs(label_first_frame[...,1]-gt_pos[0,1]),axis=-1)
    min_2 = np.min(abs(label_second_frame[...,1]-gt_pos[1,1]),axis=-1)


    # take the better label
    if(min_1<min_2):
        start_idx =1
        idx_found = np.argmin(abs(label_first_frame[...,1]-gt_pos[0,1]),axis=-1)
        radar_est.append(label_first_frame[idx_found,:])
    else:
        start_idx =2
        idx_found = np.argmin(abs(label_second_frame[...,1]-gt_pos[1,1]),axis=-1)
        radar_est.append(label_first_frame[idx_found,:])
        radar_est.append(label_second_frame[idx_found,:])
    # Overwrite if this is useless first
    if (radar_est[0][1]==-37.5):
        radar_est = np.zeros((5,2))
        return radar_est

    # Track over sequence
    for id_seq in range(start_idx,len(radar_labels)):
        # Use the current one to get the correct distance
        curr_labels = np.array( [(256-radar_labels[id_seq,:,1])/(2*RANGE_RES),ang_lim*(radar_labels[id_seq,:,0]-32)/64]).T

        # Match closest euclidian distance (i know its not correct but lets do it like it)
        idx_found = np.argmin(np.linalg.norm(curr_labels-radar_est[-1],axis=-1),axis=-1)

        # If the next one is not within a certain reach
        if(np.min(np.linalg.norm(curr_labels-radar_est[-1],axis=-1),axis=-1)<20):
            # Append
            radar_est.append(curr_labels[idx_found,:])
        else:
            radar_est.append(radar_est[-1])


    # Return distance and ranges
    return np.array(radar_est)


# Match Camera
## Fix that the batch size dimension is there
def match_camera_seq(cam_labels,gt_pos):
    # Empty arrays
    camera_target =[]

    # # Use FoV to get the angle
    angle_camera = (np.sign((cam_labels[...,1]-0.5))*np.sqrt(((cam_labels[...,1]-0.5)*640)**2 + ((cam_labels[...,2]-0.5)*480)**2)*FoV/np.sqrt(640**2+480**2))
    angle_camera[angle_camera==-55] = -200
    # Match closest angle (AS angle is the most important here anyhow)
    min_1 = np.min(abs(angle_camera[0,...]-gt_pos[0,1]),axis=-1)
    min_2 = np.min(abs(angle_camera[1,...]-gt_pos[1,1]),axis=-1)

    # take the better label
    if(min_1<min_2):
        start_idx =1
        idx_found = np.argmin(abs(angle_camera[0,...]-gt_pos[0,1]),axis=-1)
        camera_target.append(cam_labels[0,idx_found,:])
    else:
        start_idx =2
        idx_found = np.argmin(abs(angle_camera[1,...]-gt_pos[1,1]),axis=-1)
        camera_target.append(cam_labels[0,idx_found,:])
        camera_target.append(cam_labels[1,idx_found,:])

    # Track over sequence
    for id_seq in range(start_idx,len(cam_labels)):
        # Match closest euclidian distance (i know its not correct but lets do it like it)
        idx_found = np.argmin(np.linalg.norm(cam_labels[id_seq,...,1:3]-camera_target[-1][1:3],axis=-1),axis=-1)

        # If the next one is not within a certain reach
        if(np.min(np.linalg.norm(cam_labels[id_seq,...,1:3]-camera_target[-1][1:3],axis=-1),axis=-1)<0.4):
            # Append
            camera_target.append(cam_labels[id_seq,idx_found,:])
        else:
            camera_target.append(camera_target[-1])

    return np.array(camera_target)


# Match Lidar
## Fix that the batch size dimension is there
def match_lidar_seq(lidar_labels,gt_pos):
    # Output Array
    lidar_est = []
    # Convert to range angle
    lidar_labels = np.moveaxis(np.array([np.linalg.norm(lidar_labels,axis=-1), np.tan(lidar_labels[...,1]/lidar_labels[...,0])]),0,-1)
    lidar_labels[...,1] =   180*lidar_labels[...,1]/np.pi
    lidar_labels[np.isnan(lidar_labels)] = 0

    # Empty arrays
    lidar_target =[]

    # Match closest angle (AS angle is the most important here anyhow)
    min_1 = np.min(abs(lidar_labels[0,...,1]-gt_pos[0,1]),axis=-1)
    min_2 = np.min(abs(lidar_labels[1,...,1]-gt_pos[1,1]),axis=-1)


    # take the better label
    if(min_1<min_2):
        start_idx =1
        idx_found = np.argmin(abs(lidar_labels[0,...,1]-gt_pos[0,1]),axis=-1)
        lidar_target.append(lidar_labels[0,idx_found,:])
    else:
        start_idx =2
        idx_found = np.argmin(abs(lidar_labels[1,...,1]-gt_pos[1,1]),axis=-1)
        lidar_target.append(lidar_labels[0,idx_found,:])
        lidar_target.append(lidar_labels[1,idx_found,:])

    # REmove if useless e.g. 10 meters apart
    if np.min((min_1,min_2))>10:
        lidar_target = np.zeros((5,2))
        return lidar_target

    # Track over sequence
    for id_seq in range(start_idx,len(lidar_labels)):
        # Match closest euclidian distance (i know its not correct but lets do it like it)
        idx_found = np.argmin(np.linalg.norm(lidar_labels[id_seq,...]-lidar_target[-1],axis=-1),axis=-1)

        # If the next one is not within a certain reach
        if(np.min(np.linalg.norm(lidar_labels[id_seq,...]-lidar_target[-1],axis=-1),axis=-1)<20):
            # Append
            lidar_target.append(lidar_labels[id_seq,idx_found,:])
        else:
            lidar_target.append(lidar_target[-1])


    return np.array(lidar_target)



# Map Activation function
def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = nn.LeakyReLU()
    elif af_name == "Sigmoid":
        act_func = nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = nn.Tanh()
    elif af_name == "Softplus":
        act_func = nn.Softplus()
    elif af_name == "Linear":
        act_func = nn.Identity()

    else:
        sys.exit("Invalid activation function")
    return act_func

def NMSE(x_hat, x):
    nmse = torch.mean(((x-x_hat)/(torch.abs(x_hat)))**2)
    return nmse

def corr(x1, x2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
    return 1-pearson

def cce(x1, x2):
    return (-(x1+1e-5).log() * x2).sum(dim=1).mean()

def defined_losses(loss):
    if loss=='MSE':
        return nn.MSELoss()
    # Default MSE
    elif loss == "L1":
        return nn.SmoothL1Loss()
    elif loss == "BCE":
        return nn.BCELoss()
    elif loss == "NMSE":
        return NMSE
    elif loss == "Triplet":
        return nn.TripletMarginLoss()
    elif loss == "CE":
        return nn.CrossEntropyLoss()
    elif loss == "CS":
        return nn.CosineEmbeddingLoss()
    elif loss == "HL":
        return nn.HuberLoss()
    elif loss == "Corr":
        return corr
    elif loss == "NLL":
        return nn.NLLLoss()
    elif loss == "CCE":
        return cce
    else:
        sys.exit("Invalid loss function")


