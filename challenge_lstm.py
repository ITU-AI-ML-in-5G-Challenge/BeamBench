#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Code for Multimodal Beam prediction challenge
Maximilian Arnold

"""

# Import standard Libraries
import numpy as np, os, time, argparse, datetime, json, glob
import utm
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import open3d as o3d
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import h5py 

# Import torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Import Model
from models.dense_model import *
from models.lstm_model import *
from models.ae_camera_model import *
from models.ae_lidar_model import *
from models.ae_radar_model import *
from models.cl_camera_model import *
from models.cl_radar_model import *
from models.mmWave_camera_model import *
from models.mmWave_radar_model import *
from models.mmWave_lidar_model import *


# Import custom libraries
from libraries.general import *


# Parser
parser = argparse.ArgumentParser(description='Test MultiModal Challenge')
# Settings for parser
parser.add_argument('--gpu_id', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--type_list', default='adapt_camera_ae_gps', type=str, # camera_mmWave
                    help='GPU ID')
parser.add_argument('--adapt', default='', type=str, # camera_mmWave
                    help='GPU ID')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--root', default='./', type=str,
                    help='root working dir')
parser.add_argument('--csv', default='ml_challenge_test_multi_modal.csv', type=str,
                    help='csv path')
parser.add_argument('--data_folder', default='../sensenet/raw_data/test/', type=str,
                    help='csv path')
parser.add_argument('--beams_shift', default=1, type=int,
                    help='Need to shift the codebook by 1 as the original data was 1..64')

# Main Loop
def main(args):
    # Settings for seed
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse arguments
    device = args.gpu_id

    ## Settings
    csv_path = args.csv

    # Read into dataframe
    df = pd.read_csv(args.data_folder + csv_path)
    n_samples = df.index.stop
    # Get relative positons
    pos1_rel_paths = df['unit2_loc_1'].values
    pos2_rel_paths = df['unit2_loc_2'].values
    pos_bs_rel_paths = df['unit1_loc'].values

    # Get all absolute paths
    pos1_abs_paths = [os.path.join(args.data_folder, path[2:]) for path in pos1_rel_paths]
    pos2_abs_paths = [os.path.join(args.data_folder, path[2:]) for path in pos2_rel_paths]
    pos_bs_abs_paths = [os.path.join(args.data_folder, path[2:]) for path in pos_bs_rel_paths]

    # no. input sequences x no. samples per seq x sample dimension (lat and lon)
    pos_input = np.zeros((n_samples, 2))
    pos_input_2 = np.zeros((n_samples, 2))
    pos_bs = np.zeros((n_samples,2))

    # Sample index load
    for sample_idx in range(n_samples):
        # unit2 (UE) positions
        pos_input[sample_idx, :] = np.loadtxt(pos1_abs_paths[sample_idx])
        pos_input_2[sample_idx, :] = np.loadtxt(pos2_abs_paths[sample_idx])
        # unit1 (BS) position
        pos_bs[sample_idx,:] = np.loadtxt(pos_bs_abs_paths[sample_idx])

    # Convert to xy
    pos_ue_cart = xy_from_latlong(pos_input)
    pos_ue_cart_2 = xy_from_latlong(pos_input_2)
    pos_bs_cart = xy_from_latlong(pos_bs)
    pos_diff = pos_ue_cart - pos_bs_cart
    pos_diff_2 = pos_ue_cart_2 - pos_bs_cart
    # Calc Radar Distance
    radar_dist = np.linalg.norm(pos_diff,axis=-1)
    radar_dist_2 = np.linalg.norm(pos_diff_2,axis=-1)

    # Get unique BS position as each of them has their own calibration values
    _, idx  = np.unique(pos_bs_cart,axis=0, return_index=True)
    pos_bs_cart_unique = pos_bs_cart[np.sort(idx),:]
    dataset_id_find = np.linspace(0,len(pos_bs_cart_unique[:,0])-1,len(pos_bs_cart_unique[:,0]),endpoint=True).astype(int)

    dataset_id = np.zeros(len(pos_bs_cart[:,0]))
    for idx in dataset_id_find:
        id_found = np.logical_and(pos_bs_cart[:,0] == pos_bs_cart_unique[idx,0], pos_bs_cart[:,1] == pos_bs_cart_unique[idx,1])
        dataset_id[id_found] = idx

    # Rotate poss_diff around the expected angle for radar
    radar_angle = np.zeros(radar_dist.shape)
    radar_angle_2 = np.zeros(radar_dist.shape)
    mean_angle_bs = [-0.5108130249839052,-0.7214024130947583,0.5930346897155352,-0.8125375604986421+np.pi/2]

    # Correct Angle by it
    for idx in range(len(pos_bs_cart_unique[:,0])):
        newX = pos_diff[dataset_id ==idx,0] * np.cos(mean_angle_bs[idx]) - pos_diff[dataset_id ==idx,1] * np.sin(mean_angle_bs[idx])
        newY = pos_diff[dataset_id ==idx,0] * np.sin(mean_angle_bs[idx]) + pos_diff[dataset_id ==idx,1] * np.cos(mean_angle_bs[idx])
        pos_diff[dataset_id ==idx,0] =newX
        pos_diff[dataset_id ==idx,1] = newY

        newX = pos_diff_2[dataset_id ==idx,0] * np.cos(mean_angle_bs[idx]) - pos_diff_2[dataset_id ==idx,1] * np.sin(mean_angle_bs[idx])
        newY = pos_diff_2[dataset_id ==idx,0] * np.sin(mean_angle_bs[idx]) + pos_diff_2[dataset_id ==idx,1] * np.cos(mean_angle_bs[idx])
        pos_diff_2[dataset_id ==idx,0] =newX
        pos_diff_2[dataset_id ==idx,1] = newY

    for idx in range(len(pos_bs_cart_unique[:,0])):
        radar_angle[dataset_id ==idx] = 180*np.arctan(pos_diff[dataset_id ==idx,0]/pos_diff[dataset_id ==idx,1])/np.pi
        radar_angle_2[dataset_id ==idx] = 180*np.arctan(pos_diff_2[dataset_id ==idx,0]/pos_diff_2[dataset_id ==idx,1])/np.pi


    # Load Model
    # Create Model based on string insides
    input_dim = 0
    rad_bool = 0
    rad_dense =0
    gps_bool = 0
    cam_bool = 0
    cam_dense = 0
    lidar_bool = 0
    lidar_dense = 0
    direct_bool =0
    if "camera_ae" in args.type_list:
        ## Here we load the best NN Config based on your decision
        f = open(args.root + '/config/camera_ae.cfg',)
        cfg_net = json.load(f)
        # Create Model
        cam_model = ae_camera_model(cfg_net,[1,3,480,640])
        cam_model = cam_model.cuda(device)
        # Load state
        cam_model.load_state_dict(torch.load(args.root + '/results/models/adapt_bb_camera_ae_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        cam_model.eval()
        cam_model = cam_model.encoder
        cam_bool=1
        input_dim +=512
    elif "camera_cl" in args.type_list:
        f = open(args.root + '/config/camera_cl.cfg',)
        cfg_net = json.load(f)
        # Create Model
        cam_model = cl_camera_model(cfg_net,[1,3,480,640],1024)
        cam_model.load_state_dict(torch.load(args.root + '/results/models/adapt_bb_camera_cl_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        cam_model = cam_model.encoder_q
        cam_model.eval()
        cam_bool=1
        input_dim +=1024
    elif "camera_mmWave" in args.type_list:
        direct_bool =1
        f = open(args.root + '/config/camera_mmWave.cfg',)
        cfg_net = json.load(f)
        # Create Model
        model = mmWave_camera_model(cfg_net,[1,3,480,640],64)
        model = model.cuda(device)
        model.load_state_dict(torch.load(args.root + '/results/models/' + args.adapt + 'camera_mmWave_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        model.eval()
        cam_bool=1
    elif "camera_dense" in args.type_list:
        cam_bool =1 
        input_dim +=2
    if "radar_ae" in args.type_list:
        f = open(args.root + '/config/radar_ae.cfg',)
        cfg_net = json.load(f)
        cam_model = ae_radar_model(cfg_net,[1,256,64])
        radar_model = radar_model.cuda(device)
        radar_model.load_state_dict(torch.load(args.root + '/results/models/adapt_bb_radar_ae_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        radar_model = radar_model.encoder
        radar_model.eval()
        rad_bool = 1
        input_dim +=512
    elif "radar_cl" in args.type_list:
        f = open(args.root + '/config/radar_cl.cfg',)
        cfg_net = json.load(f)
        radar_model = cl_radar_model(cfg_net,[1,256,64],1024)
        radar_model = radar_model.cuda(device)
        radar_model.load_state_dict(torch.load(args.root + '/results/models/adapt_bb_radar_cl_' + str(0)  + '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        radar_model = radar_model.encoder_q
        radar_model.eval()
        rad_bool = 1
        input_dim +=1024
    elif "radar_mmWave" in args.type_list:
        direct_bool =1
        f = open(args.root + '/config/radar_mmWave.cfg',)
        cfg_net = json.load(f)
        # Create Model
        model = mmWave_radar_model(cfg_net,[1,256,64],64)
        model = model.cuda(device)
        model.load_state_dict(torch.load(args.root + '/results/models/' + args.adapt + 'radar_mmWave_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        model.eval()
        rad_bool =1
    elif "radar_dense" in args.type_list:
        rad_bool =1 
        input_dim +=2

    if "lidar_ae" in args.type_list:
        f = open(args.root + '/config/lidar_ae.cfg',)
        cfg_net = json.load(f)
        lidar_model = ae_lidar_model(cfg_net,[5,100,3])
        lidar_model = lidar_model.cuda(device)
        lidar_model.load_state_dict(torch.load(args.root + '/results/models/adapt_bb_lidar_ae_' +  str(0) + '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        lidar_model.eval()
        lidar_model = lidar_model.encoder
        lidar_bool = 1
        input_dim +=512
    elif "lidar_mmWave" in args.type_list:
        direct_bool =1
        f = open(args.root + '/config/lidar_mmWave.cfg',)
        cfg_net = json.load(f)
        # Create Model
        model = mmWave_lidar_model(cfg_net,[5,100,3],64)
        model = model.cuda(device)
        model.load_state_dict(torch.load(args.root + '/results/models/' + args.adapt + 'lidar_mmWave_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        model.eval()
        lidar_bool=1
    elif "lidar_dense" in args.type_list:
        lidar_bool =1 
        input_dim +=2
    if "gps_dense" in args.type_list:
        f = open(args.root + '/config/gps_dense.cfg',)
        cfg_net = json.load(f)
        # Create Model
        model = dense_model(cfg_net,2,64) # Use here hidden size
        model.load_state_dict(torch.load(args.root + '/results/models/' + args.adapt + 'gps_dense_' +  str(0)+ '_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        model.eval()
        gps_bool = 1
    elif "gps" in args.type_list:
        input_dim +=2
        gps_bool = 1

    len_seq =5 
    # Load Model
    if (direct_bool==0):
        f = open(args.root + '/config/lstm_model.cfg')
        cfg_net = json.load(f)
        model = lstm_model(cfg_net,input_dim,64)
        model = model.cuda(device)
        # Load state if they are single
        model.load_state_dict(torch.load(args.root + '/results/models/lstm_' + args.adapt  + args.type_list + '_0_' + str(args.seed) + '.pth',map_location='cuda:' + str(device)))
        model.eval()


    # Load Preprocessed matchings
    f_test = h5py.File('deepsense_6g_test_match.hdf5','r')

    # Go over all datasets and load the data
    beam_topk = []
    for sample_idx in tqdm(range(n_samples)):
        imgs = []
        radars = []
        gt_pos_arr = []
        lidarr = []
        for idx_seq in range(len_seq):
            # Read image
            img_abs_path = os.path.join(args.data_folder, df['unit1_rgb_'+str(5)][sample_idx])
            img = plt.imread(img_abs_path)
    
            # Resize
            img  = np.moveaxis(cv2.resize(img, [640,480]),-1,0)
            imgs.append(img)
    
            # Load Radar
            radar_data = np.load(os.path.join(args.data_folder, df['unit1_radar_'+str(5)][sample_idx]))
    
            # Parse Radar data
            radar_range_ang_data = range_angle_map(radar_data)#[first_range_sample:last_range_sample]/np.max(range_angle_map(radar_data)[first_range_sample:last_range_sample])
            radar = 255*np.flip(radar_range_ang_data,axis=0)
            radars.append(radar)
            
            # Stack GT Pos
            gt_pos_arr.append(np.array([radar_dist[sample_idx],radar_angle[sample_idx]]).astype(np.float32))
    
            # Get lidar
            if(lidar_bool):
                # Lidar unit1_lidar_1
                lidar_path = os.path.join(args.data_folder, df['unit1_lidar_'+str(5)][sample_idx])
                cloud = o3d.io.read_point_cloud(lidar_path)
                points = np.asarray(cloud.points)
                points = points[points[:,-1]>-1,:]
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                labels = np.array(cloud.cluster_dbscan(eps=1, min_points=10))
                max_label = labels.max()
                colors = plt.get_cmap("tab20")(labels / (max_label  if max_label > 0 else 1))
                colors[labels < 0] = 0
                cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
                # Sweep over the labels and find the point cloud matching to the
                cloud_center = []
                for idx_cloud in np.arange(max_label):
                    nb_points = len(points[labels==idx_cloud,0:2])
                    if nb_points >150:
                        labels[labels==idx_cloud]=0
                    else:
                        cloud_center.append(np.mean(points[labels==idx_cloud,:],axis=0))
                lidar = np.zeros((1,3,100))
                lidar[0,:,0:len(cloud_center)] = np.array(cloud_center).T/100
                lidarr.append(lidar)
        # Get the current matched
        radar_labels = torch.from_numpy(f_test['radar_match'][sample_idx,...])    
        camera_bb = torch.from_numpy(f_test['camera_match'][sample_idx,...])
        lidar_labels = torch.from_numpy(f_test['lidar_match'][sample_idx,...])
        camera = torch.from_numpy(np.array(imgs).astype(np.float32))[None,...]/255
        radar = torch.from_numpy(np.array(radars).astype(np.float32))[None,...]
        gt_pos =torch.from_numpy(np.array(gt_pos_arr))[None,...]
        lidar =torch.from_numpy(np.array(lidarr).astype(np.float32))[None,...]

        if (direct_bool==0):
            Input_full = torch.empty(0).cuda(device)
            for idx_seq in range(len_seq): # sequence id from 0 ...4
                #print(idx_seq)
                Input = torch.empty(0).cuda(device)
                if(rad_bool):
                    if(rad_dense):
                        Input = torch.cat((Input,radar_labels[0:2,idx_seq,...].cuda(device)),-1)
                    else:
                        Input = torch.cat((Input,radar_model(radar[:,idx_seq,...].cuda(device))),-1)
                if(cam_bool):
                    if(cam_dense):
                        Input = torch.cat((Input,camera_bb[0:2,idx_seq,...].cuda(device)),-1)
                    else:
                        Input = torch.cat((Input,cam_model(camera[:,idx_seq,...].cuda(device))),-1)
                if(lidar_bool):
                    if(lidar_dense):
                        Input = torch.cat((Input,lidar_label[0:2,idx_seq,...].cuda(device)),-1)
                    else:
                        Input = torch.cat((Input,lidar_model(lidar[...,idx_seq].cuda(device))),-1)
                if(gps_bool):
                    # If sequence smaller 2 we can use gps else its not available
                    if(idx_seq<2):
                        Input = torch.cat((Input,gt_pos[:,idx_seq,...].cuda(device)),-1)
                    else:
                        # Just add zeros to the input
                        Input = torch.cat((Input,torch.ones((1,2)).cuda(device)),-1)
                if(len(Input_full) ==0):
                    Input_full = Input[:,None,:]
                else:
                    Input_full = torch.cat((Input_full,Input[:,None,:]),1)

            # Test Model
            beam_topk.append(torch.topk(model(Input_full).cpu().detach(),3)[1].numpy())
        else:
            if(rad_bool):
                output = model(radar.cuda(device))
            if(cam_bool):
                output = model(img.cuda(device))
            if(gps_bool):
                output = model(gt_pos.cuda(device))

            # Test Model
            beam_topk.append(torch.topk(output.cpu().detach(),3)[1].numpy())


    # Save predictions to csv
    save_pred_to_csv(args.beams_shift + np.array(beam_topk)[:,0,:]+1,target_csv=args.root + '/results/topk/fusion_' + args.adapt + args.type_list + '_' + str(args.seed) + '.csv')



# If called via script then just this
if __name__ == '__main__':
    # Parse ARgs
    args = parser.parse_args()
    # Run Main
    main(args)
