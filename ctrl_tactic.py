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


#%% Setup hyperparameters

# === Saber Flocking =====
a = 0.5
b = 0.5
c = np.divide(np.abs(a-b),np.sqrt(4*a*b)) 
eps = 0.1
#eps = 0.5
h = 0.9
pi = 3.141592653589793

# gains
c1_a = 2                # lattice flocking
c2_a = 2*np.sqrt(2)
c1_b = 1                # obstacle avoidance
c2_b = 2*np.sqrt(1)
c1_g = 3                # target tracking
c2_g = 2*np.sqrt(3)

# # === Encirclement+ ===
c1_d = 2                # encirclement 
c2_d = 2*np.sqrt(2)

#%% Some function that are used often
# ---------------------------------

def regnorm(z):
    norm = np.divide(z,np.linalg.norm(z))
    return norm

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig
    
def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h
 
def phi_a(q_i, q_j, r_a, d_a): 
    z = sigma_norm(q_j-q_i)        
    phi_a = rho_h(z/r_a) * phi(z-d_a)    
    return phi_a
    
def phi(z):    
    phi = 0.5*((a+b)*sigma_1(z+c)+(a-b))    
    return phi 
        
def a_ij(q_i, q_j, r_a):        
    a_ij = rho_h(sigma_norm(q_j-q_i)/r_a)
    return a_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b
 
def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out


    
#%% Tactic Command Equations 
# ------------------------
def commands(states_q, states_p, obstacles, walls, r, d, r_prime, d_prime, targets, targets_v, targets_enc, targets_v_enc, swarm_prox, tactic_type, centroid, escort):   
    
    # initialize 
    u_int = np.zeros((3,states_q.shape[1]))     # interactions
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation
    u_enc = np.zeros((3,states_q.shape[1]))     # encirclement 
    cmd_i = np.zeros((3,states_q.shape[1]))     # store the commands
        
    # if doing Reynolds, reorder the agents 
    if tactic_type == 'reynolds':
        distances = reynolds_tools.order(states_q)
   
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


        # if structured swarming
        if tactic_type == 'circle':    
            # Encirclement term (phi_delta)
            # ----------------------------    
            u_enc[:,k_node] = encircle_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
            
        
        # if structured swarming
        if tactic_type == 'lemni':    
            # Encirclement term (phi_delta)
            # ----------------------------    
            u_enc[:,k_node] = lemni_tools.compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node)
  

        # Conditional commands
        # ----------------------------------------------         
        if tactic_type == 'saber':
            cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
        elif tactic_type == 'reynolds':
            cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node] # adds the saber obstacle avoidance 
        elif tactic_type == 'circle':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 
        elif tactic_type == 'lemni':
            cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 

    cmd = cmd_i    
    
    return cmd




