#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Plotting different results """
# Import Libraries
import numpy as np, os, time, argparse, datetime, json
import pandas as pd

# Custom libraries
from libraries.general import *

# Parser
parser = argparse.ArgumentParser(description='Test MultiModal Challenge')
# Settings for parser
parser.add_argument('--gpu_id', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--type_list', default='lidar_ae_camera_ae_radar_cl_gps', type=str, # camera_mmWave
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


def func(x, a, b,c,d):
    y = a*x**3 +b*x**2+c*x + d
    return y

scene_id_arr = [0,3,1,2]
# Main Loop
def main(args):
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

    # GT
    current_range = radar_dist
    current_angle = radar_angle

    # Normalize
    mean_angle = np.loadtxt(args.root + '/results/models/classic_angle.npy')
    for ids in range(len(mean_angle)):
        current_angle[scene_id_arr[ids]==dataset_id] = current_angle[scene_id_arr[ids]==dataset_id]- mean_angle[ids]

    # Load Alpha
    alpha = np.loadtxt(args.root + '/results/models/classic.npy')
    predicted_beams = func(current_angle,alpha[0],alpha[1],alpha[2],alpha[3])
    predicted_beams = np.array([predicted_beams, predicted_beams+1,predicted_beams+2]).T

    # Correct per Scenario an average offset
    corr = np.loadtxt(args.root + '/results/models/classic_corr.npy')
    for ids in range(len(corr)):
        predicted_beams[scene_id_arr[ids]==dataset_id] = predicted_beams[scene_id_arr[ids]==dataset_id]+ corr[ids]

    # Save predictions to csv
    save_pred_to_csv(args.beams_shift + predicted_beams,target_csv=args.root + '/results/topk/classical.csv')


 # If called via script then just this
if __name__ == '__main__':
    # Parse ARgs
    args = parser.parse_args()
    # Run Main
    main(args)



