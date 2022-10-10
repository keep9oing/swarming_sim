#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the control inputs for various swarming strategies 

Created on Mon Jan  4 12:45:55 2021

@author: tjards

"""

import numpy as np
import reynolds_tools
import saber_tools
import encirclement_tools as encircle_tools
import lemni_tools
import staticShapes_tools as statics
import starling_tools
#params = np.zeros((4,3))  # store the parameters commands



#%% Setup hyperparameters

eps = 0.5
    
#%% Tactic Command Equations 
# ------------------------
def commands(states_q, states_p, obstacles, walls, r, d, r_prime, d_prime, targets, targets_v, targets_enc, targets_v_enc, swarm_prox, tactic_type, centroid, escort, params):   
    
    # initialize 
    u_int = np.zeros((3,states_q.shape[1]))     # interactions
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation
    u_enc = np.zeros((3,states_q.shape[1]))     # encirclement 
    u_statics = np.zeros((3,states_q.shape[1])) # statics
    cmd_i = np.zeros((3,states_q.shape[1]))     # store the commands
        
    # if doing Reynolds, reorder the agents 
    if tactic_type == 'reynolds':
        distances = reynolds_tools.order(states_q)
        # if doing Reynolds, reorder the agents 
    #elif tactic_type == 'starling':
    #    distances = starling_tools.order(states_q)
        
   
    # for each vehicle/node in the network
    for k_node in range(states_q.shape[1]): 
                 
        # Reynolds Flocking
        # ------------------
        if tactic_type == 'reynolds':
           # compute command 
           cmd_i[:,k_node] = reynolds_tools.compute_cmd(targets, centroid, states_q, states_p, k_node, r, r_prime, escort, distances)
           

        # Saber Flocking
        # ---------------                                
        if tactic_type == 'saber':
               
            # Lattice Flocking term (phi_alpha)
            # ---------------------------------
            u_int[:,k_node] = saber_tools.compute_cmd_a(states_q, states_p, targets, targets_v, k_node, r, d, r_prime, d_prime)    

            # Navigation term (phi_gamma)
            # ---------------------------
            u_nav[:,k_node] = saber_tools.compute_cmd_g(states_q, states_p, targets, targets_v, k_node)
              
            
        # Obstacle Avoidance term (phi_beta)
        # ---------------------------------   
        u_obs[:,k_node] = saber_tools.compute_cmd_b(states_q, states_p, obstacles, walls, k_node, r_prime, d_prime)


        # Encirclement term (phi_delta)
        # ---------------------------- 
        if tactic_type == 'circle':       
            u_enc[:,k_node] = encircle_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
        # Lemniscatic term (phi_lima)
        # ---------------------------- 
        if tactic_type == 'lemni':    
            u_enc[:,k_node] = lemni_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
        
        if tactic_type == 'statics':
            u_statics[:,k_node] = statics.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
        # Starling
        # --------
        if tactic_type == 'starling':
           # compute command 
           #cmd_i[:,k_node] = starling_tools.compute_cmd(targets, centroid, states_q, states_p, k_node, r, r_prime, escort)
           cmd_i[:,k_node], params = starling_tools.compute_cmd(targets, centroid, states_q, states_p, k_node, escort, params, 0.02)
        
        
        
        # Mixer
        # -----         
        if tactic_type == 'saber':
            cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
        elif tactic_type == 'reynolds':
            cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node] # adds the saber obstacle avoidance 
        elif tactic_type == 'circle':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 
        elif tactic_type == 'lemni':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node]
        elif tactic_type == 'statics':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_statics[:,k_node]
        elif tactic_type == 'starling':
            cmd_i[:,k_node] = cmd_i[:,k_node] 

    cmd = cmd_i    
    
    return cmd, params




