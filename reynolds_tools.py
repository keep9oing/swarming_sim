#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:12:59 2021

@author: tjards

This program implements Reynolds Rules of Flocking ("boids")

"""

import numpy as np

# Hyperparameters
# ----------------

cd_1 = 0.4              # cohesion
cd_2 = 0.3              # alignment
cd_3 = 0.8              # separation
cd_4 = 0                # navigation (default 0)
maxu = 10               # max input (per rule)
maxv = 100              # max v
far_away = 500          # when to go back to centroid
agents_min_coh = 3      # min number of agents
mode_min_coh = 1        # enforce min # of agents (0 = no, 1 = yes)
cd_escort = 0.5         # gain to use for escort

# Some useful functions
# ---------------------

def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out

def order(states_q):
    distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
    
    # to find the radius that includes min number of agents
    if mode_min_coh == 1:
        slide = 0
        for k_node in range(states_q.shape[1]):
            #slide += 1
            for k_neigh in range(slide,states_q.shape[1]):
                if k_node != k_neigh:
                    distances[k_node,k_neigh] = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
    return distances 

# Compute commands
# ----------------

def compute_cmd(targets, centroid, states_q, states_p, k_node, r, r_prime, escort, distances):

    # Reynolds Flocking
    # ------------------ 
    
    #initialize commands 
    u_coh = np.zeros((3,states_q.shape[1]))  # cohesion
    u_ali = np.zeros((3,states_q.shape[1]))  # alignment
    u_sep = np.zeros((3,states_q.shape[1]))  # separation
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation
    #distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
    cmd_i = np.zeros((3,states_q.shape[1])) 
    
    #initialize for this node
    temp_total = 0
    temp_total_prime = 0
    temp_total_coh = 0
    sum_poses = np.zeros((3))
    sum_velos = np.zeros((3))
    sum_obs = np.zeros((3))
    u_coh = np.zeros((3,states_q.shape[1]))  # cohesion
    u_ali = np.zeros((3,states_q.shape[1]))  # alignment
    u_sep = np.zeros((3,states_q.shape[1]))  # separation
    
    # adjust cohesion range for min number of agents 
    if mode_min_coh == 1:
        
        # make sure the number of vehicles is bigger than the min number of agents 
        if distances.shape[0] < agents_min_coh+2:
            raise Exception('There are an insufficient number of agents for the cohesion mode selected. Minimum number of agents for mode ',agents_min_coh ,' is ', agents_min_coh+2, ' and you have selected ', distances.shape[0] )
    
        r_coh = 0
        #agents_min_coh = 5
        node_ranges = distances[k_node,:]
        node_ranges_sorted = np.sort(node_ranges)
        r_coh_temp = node_ranges_sorted[agents_min_coh+1]
        r_coh = r_coh_temp
        #print(r_coh)
    else:
        r_coh = r
          
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
        # except for itself (duh):
        if k_node != k_neigh:
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
            
            if dist < 0.1:
                print('collision at agent: ', k_node)
                continue
    
            # if it is within the alignment range
            if dist < np.maximum(r,r_coh):
     
                # count
                temp_total += 1                        
     
                # sum 
                #sum_poses += states_q[:,k_neigh]
                sum_velos += states_p[:,k_neigh]
    
            # if within cohesion range 
            if dist < np.maximum(r,r_coh):
                
                #count
                temp_total_coh += 1
                
                #sum
                sum_poses += states_q[:,k_neigh]
    
    
            # if within the separation range 
            if dist < r_prime:
                
                # count
                temp_total_prime += 1
                
                # sum of obstacles 
                sum_obs += -(states_q[:,k_node]-states_q[:,k_neigh])/(dist**2)                        
    
    # norms
    # -----
    norm_coh = np.linalg.norm(sum_poses)
    norm_ali = np.linalg.norm(sum_velos)
    norm_sep = np.linalg.norm(sum_obs)
      
    if temp_total != 0:
        
        # Cohesion
        # --------
        if norm_coh != 0:
            #temp_u_coh = (maxv*np.divide(((np.divide(sum_poses,temp_total) - states_q[:,k_node])),norm_coh)-states_p[:,k_node])
            temp_u_coh = (maxv*np.divide(((np.divide(sum_poses,temp_total_coh) - states_q[:,k_node])),norm_coh)-states_p[:,k_node])
            u_coh[:,k_node] = cd_1*norm_sat(temp_u_coh,maxu)
            #print(temp_total_coh)
        
        # Alignment
        # ---------
        if norm_ali != 0:                 
            temp_u_ali = (maxv*np.divide((np.divide(sum_velos,temp_total)),norm_ali)-states_p[:,k_node])
            u_ali[:,k_node] = cd_2*norm_sat(temp_u_ali,maxu)
    
    if temp_total_prime != 0 and norm_sep != 0:
            
        # Separtion
        # ---------
        temp_u_sep = (maxv*np.divide(((np.divide(sum_obs,temp_total_prime))),norm_sep)-states_p[:,k_node]) 
        u_sep[:,k_node] = -cd_3*norm_sat(temp_u_sep,maxu)
                
    # Tracking
    # --------            
    # if far away
    if np.linalg.norm(centroid.transpose()-states_q[:,k_node]) > far_away:
        cd_4 = 0.05
    else:
        cd_4 = 0
    
    if escort == 1:
        cd_4 = cd_escort
        temp_u_nav = (targets[:,k_node]-states_q[:,k_node])
    else:
        temp_u_nav = (centroid.transpose()-states_q[:,k_node])
    u_nav[:,k_node] = cd_4*norm_sat(temp_u_nav,maxu)
    
    cmd_i[:,k_node] = u_coh[:,k_node] + u_ali[:,k_node] + u_sep[:,k_node] + u_nav[:,k_node] 
    
    return cmd_i[:,k_node]
  