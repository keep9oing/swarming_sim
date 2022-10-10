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

#cd_coh =  0.2     # cohesion
#cd_ali =  0.2     # alignment
#cd_sep =  0.2     # separation
#maxu =  10     # max input (per rule)  note: dynamics *.evolve_sat must be used for constraints
#maxv =  100    # max v                 note: dynamics *.evolve_sat must be used for constraints
v_o =   10      # cruise speed
m   =   0.08    # agent mass (could be different per)
tau =   0.2     # relaxation time (tunable)
del_u = 0.1    # reaction time (to new neighbours)
#del_t = 0.02   # integration step 
s =     0.1*del_u # interpolation factor
#R_i =   5           # interaction radius (initialize)
R_max = 100         # interation radius, max
#n_i =   0           # topical range count (initialize)
n_c =   6.5         # "topical range" (i.e. min number of agents to pay attention to)
r_sep = 10 #4       # separation radius
r_h =   0.2     # hard sphere (ignore things too close for cohesion)
r_roost = 50    # radius of roost
w_s = 1 #1         # weighting factor separation force 
w_c = 0.7 #1         # weighting factor for cohesion
w_a = 0.2 # 0.5        # weighting factor for alignment
w_roost_h = 0.25 # 0.01 # weighting factor for horizontal attraction to roost
w_roost_v = 0.1 #0.2 # weighting factor for vertical attraction to roost
w_rand = 0.01   # weight factor of random disturbances
C_c = 0.35      # critical centrality below which member is interior to roost

alpha = 0.5     # between 0 and 1. modulates how tightly swarms converges into target. 0 is very loose, 1 is very tight 
#r_internal =

eps = 0.00001

sigma = np.sqrt(np.divide(np.square(r_sep-r_h),4.60517)) #std dev of the gaussion set, such that at that separation zone, near zero
sigma_sqr = np.square(sigma)

# params is imported, and stores important values, namely the last interaction radius: [R_i(previous), 0]


# Some useful functions
# ---------------------

# computes a unit vector in the direction of the agent velo (i.e. forward)
def unit_vector_fwd(velos):
    vector_out = np.divide(velos,np.linalg.norm(velos))
    return vector_out     # output is a unit vector
    
# brings agent back to cruised speed, v_0 after deviating
def to_cruise(m, tau, v_o, v_i, e_x):
    f_out = np.divide(m,tau)*(v_o-v_i)*e_x
    return f_out  # output is a force 
    
# topical interaction distance 
def update_interaction(s,R_i,R_max,n_i,n_c):
    R_new = (1-s)*R_i+s*(R_max-R_max*np.divide(n_i,n_c))
    return R_new
    
# gaussian set used for separation term
def gaussian_set(d_ij,r_h, sigma):
    if d_ij <= r_h:
        return 1
    else:
        return np.exp(-np.divide(np.square(d_ij-r_h),sigma_sqr))
    
# find the rotation matrix between two vectors
# def find_R(vec1, vec2):
#     # vec1: "source" vector
#     # vec2: "destination" vector
#     # returns R_3 such that R_3*vec1 = vec2

#     a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
#     R_3 = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
#     return R_3

    


# Compute commands for Starling Flocking
# --------------------------------------

# this is run for each node
def compute_cmd(targets, centroid, states_q, states_p, k_node, escort, params, Ts):

     
    #initialize commands 
    # ------------------
    
    u_coh = np.zeros((3,1))  # cohesion
    f_coh = np.zeros((1,3))
    u_ali = np.zeros((3,1))  # alignment
    f_ali = np.zeros((1,3))   
    u_sep = np.zeros((3,1))  # separation
    f_sep = np.zeros((1,3))
    
    u_roost_h = np.zeros((3,1))  # roosting (horizontal)
    f_roost_h = np.zeros((1,3))
    u_roost_v = np.zeros((3,1))  # roosting (vertical)
    f_roost_v = np.zeros((1,3)) 
    
    u_nav = np.zeros((3,1))  # navigation
    
    cmd_i = np.zeros((3,1)) 
    params_i = np.zeros((1,4))
    if params[0,0] == 0:
        R_i =   5           # interaction radius (initialize)
        params[0,:] = R_i # for the first time
    
    
    # pull parameters for this node
    params_i = params[:,k_node]
    
    # interaction parameters
    R_i = params_i[0] # interaction range (previous)
    n_i = params_i[1] # number of agents in range (previous)
    n_counter = 0     # number of agents in range (initialize for this time)
    params_i[2] += 1  # counter to update range (slower than sample time)
    
    # centrality
    C_i = params_i[3]
    n_counter_centrality = 0 # number of agents in range for measuring centrality, nominally 2xR_i (initialize for this time)

    # alignment (initialize)
    f_ali = -unit_vector_fwd(states_p[:,k_node])



    # SOCIAL BEHAVIOURS
    # =================

    # update interaction range, if it's time
    # --------------------------------------
    if params_i[2] >= round(del_u/0.02,0)+1:
        # expand the range
        R_i = update_interaction(s,params_i[0],R_max,params_i[1],n_c)
        # reset the counter
        params_i[2] = 0
        
           
    # search through each neighbour
    # -----------------------------
    for k_neigh in range(states_q.shape[1]):
        
        # except for itself
        if k_node != k_neigh:
             
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_neigh]-states_q[:,k_node])
            #blind spot?
            
        
            # if the neighbour is within the range for computing centrality
            if dist <= 2*R_i: #yes, centrality is measured in a bigger range
            
                # increment the counter
                n_counter_centrality += 1
                
                # compute centrality
                C_i += np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist)
            
                        
            # if the neighhour is within the interaction range (add blind spot later)
            if dist <= R_i and n_counter < n_c:
                
                # increment the counter
                n_counter += 1
                
                # compute the separation force
                # ----------------------------
                f_sep += gaussian_set(dist, r_h, sigma)*np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist) 
            
            
                # compute the cohesion force 
                # -------------------------- 
                if dist > r_h:
                    f_coh += np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist)
                    
                # compute the alignment force
                # ---------------------------
                f_ali += unit_vector_fwd(states_p[:,k_neigh])
                
             
    # compute consolidated commands 
    if n_counter_centrality > 0:
        # update the centrality for this node
        C_i = np.divide(np.linalg.norm(C_i),n_counter_centrality)
        # save 
        params[3,k_node] = C_i
        
    if n_counter > 0:
        u_sep = -m*np.divide(w_s,n_counter)*f_sep.reshape((3,1))
        u_coh = m*C_i*np.divide(w_c,n_counter)*f_coh.reshape((3,1))
        u_ali = m*w_a*np.divide(f_ali,np.linalg.norm(f_ali)).reshape((3,1))
        
        
    # ROOSTING BEHAVIOURS
    # ===================
    
    # define a unit vector in y direction
    #unit_vector_x = np.array([1,0,0])
    #unit_vector_f = unit_vector_fwd(states_p[:,k_node]
    # find rotation of body
    #Rot = find_R(unit_vector_x, unit_vector_f))
    unit_vector_hor = np.array([1,1,0])
    unit_vector_ver = np.array([0,0,1])
    #unit_vector_x = np.array([1,0,0])
    #unit_vector_y = np.array([0,1,0])
    
    
    # find the Rotation matrix
    #Rot = find_R(unit_vector_x, unit_vector_fwd(states_p[:,k_node])) # from x to forward
    # find the vector to bank (not sure of sign here)
    #bank_to_roost = np.dot(Rot,unit_vector_y) 
    
    # find the vector for the forward direction (2D)
    unit_fwd_2D = unit_vector_fwd(states_p[0:2,k_node])
    # rotate it 90 degrees counterclockwise
    unit_bank_2D = np.array([-unit_fwd_2D[1],unit_fwd_2D[0]])
    # determine sign 
    # if moves farther from target
    if np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))  +  unit_bank_2D.reshape((1,2))) >= np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))):
        # switch the sign (clockwise rotation)
         unit_bank_2D = np.array([unit_fwd_2D[1],-unit_fwd_2D[0]])
    # compute unit vector towards target
    unit_to_target = np.divide(targets[0:3,k_node].reshape((1,3)) - states_q[0:3,k_node].reshape((1,3)),np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[0:3,k_node].reshape((1,3))))
    #unit_away_target = np.divide(states_q[0:3,k_node].reshape((1,3))-targets[0:3,k_node].reshape((1,3)),np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[0:3,k_node].reshape((1,3))))
    unit_to_target_2D = np.divide(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2)),np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))))
    
    
    # compute dot product of fwd and target direction (measure of how much it is pointing inwards)
    #proj_to_target = np.dot(unit_fwd_2D.reshape((1,2)), unit_to_target[:,0:2].reshape((2,1))).ravel()
    #proj_to_target = np.dot(unit_vector_fwd(states_p[0:3,k_node]).reshape((1,3)), unit_to_target[:,0:3].reshape((3,1))).ravel()
    proj_to_target = np.dot(unit_vector_fwd(states_p[0:3,k_node]).reshape((1,3)), -unit_to_target[:,0:3].reshape((3,1))).ravel()
    # above is actually projection away from target
    #proj_to_target = np.dot(unit_vector_fwd(states_p[0:2,k_node]).reshape((1,2)), -unit_to_target[:,0:2].reshape((2,1))).ravel()
    proj_to_target_2D = np.dot(unit_vector_fwd(states_p[0:2,k_node]).reshape((1,2)), -unit_to_target[:,0:2].reshape((2,1))).ravel() 

    
    # horizontal
    #f_roost_h = np.divide((targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node])*unit_vector_hor,np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node]))
    #f_roost_h = np.divide((targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node])*bank_to_roost,np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node]))
    
    #np.divide((targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node]),np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node]))
    
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)
    #f_roost_h = np.divide((targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node].reshape((1,3)))*unit_vector_hor,np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[:,k_node].reshape((1,3))))
    
    # travis: this sort of works, but I want to make it a function of how much it points to target
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)
    # make the force less strong the more it points towards target * more the farther away
    
    
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(proj_to_target)
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(alpha + (1-alpha)*proj_to_target)
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(proj_to_target)
    sign = np.sign(proj_to_target)
    #f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(sign*alpha + sign*(1-alpha)*proj_to_target)
    f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(sign*alpha + sign*(1-alpha)*proj_to_target_2D)
    
    u_roost_h = -m*w_roost_h*f_roost_h.reshape((3,1)) 

    
    #f_roost_v = (targets[2,k_node] - states_q[:,k_node])*unit_vector_ver
    f_roost_v =  (targets[2,k_node] - states_q[2,k_node])*unit_vector_ver
    u_roost_v = m*w_roost_v*f_roost_v.reshape((3,1))
    
        
    # CONSOLIDATION
    # =============
        
    params[0,k_node] = R_i
    params[1,k_node] = n_counter
    params[2,k_node] = params_i[2]
    params[3,k_node] = C_i
    
    cmd_i = u_coh + u_ali + u_sep + u_roost_h + u_roost_v
    
    return cmd_i.ravel(), params
  

# # for testing (this will move outside)


# Ts = 0.02


# targets = np.array([[-0.85  , -0.85  , -0.85  , -0.85  , -0.85  , -0.85  , -0.85  ],
#        [-0.625 , -0.625 , -0.625 , -0.625 , -0.625 , -0.625 , -0.625 ],
#        [15.0375, 15.0375, 15.0375, 15.0375, 15.0375, 15.0375, 15.0375],
#        [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
#        [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
#        [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]])

# centroid = np.array([[ 0.24156813],
#        [15.48540523],
#        [25.25880888]])

# states_q = np.array([[ 14.19933613,  -7.26692   ,  -4.49649376,  22.67654289,
#         -24.59333582, -17.39147982,  18.5633273 ],
#        [ 12.82773581,  11.17464717,  21.53516725,  24.88439621,
#           2.19520852,  19.72218196,  16.05849971],
#        [ 12.12200255,   6.97796425,  43.79418737,  12.64483405,
#          19.54808908,  35.22525037,  46.49933449]])

# states_p= np.array([[-9.37112267e-03,  7.86833964e-03,  4.63448667e-03,
#          3.09482248e-03,  9.15889935e-03, -2.11202080e-03,
#          2.59599714e-03],
#        [ 7.35657020e-04,  9.67134221e-03, -2.60620224e-03,
#         -6.36139638e-05,  5.62814630e-03, -7.36876491e-03,
#          7.22135981e-03],
#        [-9.65828014e-03, -2.88773981e-03, -2.28121461e-03,
#          3.99074161e-03, -6.76019095e-03,  6.10657200e-03,
#          7.33301538e-03]])

# escort = 0

# k_node = 0

# cmd = np.zeros((3,states_q.shape[1]))     # store the commands
# params = np.zeros((4,states_q.shape[1]))  # store the parameters commands 
# params[0,:] = R_i



# cmd[:,k_node], params = compute_cmd(targets, centroid, states_q, states_p, k_node, escort, params, Ts)
    
  
    