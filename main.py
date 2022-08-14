#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies including:
    
    - Reynolds rules of flocking ("boids")
    - Olfati-Saber flocking
    - Dynamic Encirclement 
    - Leminiscatic Arching
    - Static Shapes

The strategies requires no human invervention once the target is selected and all agents rely on local knowledge only. 
Each vehicle makes its own decisions about where to go based on its relative position to other vehicles

Created on Tue Dec 22 11:48:18 2020

@author: tjards

"""

#%% Import stuff
# --------------

#from scipy.integrate import ode
import numpy as np
import animation 
import dynamics_node as node
import tools as tools
import encirclement_tools as encircle_tools
import ctrl_tactic as tactic 
import pickle 
import quaternions as quat
import lemni_tools 
import swarm_metrics 
import staticShapes_tools as statics
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#plt.style.use('classic')
plt.style.use('default')
#plt.style.available
#plt.style.use('Solarize_Light2')

#%% Setup Simulation
# ------------------
Ti      = 0         # initial time
Tf      = 30        # final time 
Ts      = 0.02      # sample time
nVeh    = 15         # number of vehicles
iSpread = 10      # initial spread of vehicles
escort  = 0         # escort/ target tracking? (0 = no, 1 = yes)
tactic_type = 'statics'     
                # reynolds = Reynolds flocking + Olfati-Saber obstacle
                # saber = Olfati-Saber flocking
                # circle = encirclement
                # lemni = dynamic lemniscate
                # statics = static shapes (not ready yet)

# speed of target
tSpeed = 0.1

# parameters for dynamic encirclement and lemniscate
r_desired = 5                                   # desired radius of encirclement [m]
ref_plane = 'horizontal'                        # defines reference plane (default horizontal)
phi_dot_d = 0.1 # 0.05 # 0.12                                # how fast to encircle
unit_lem = np.array([1,0,0]).reshape((3,1))     # sets twist orientation (i.e. orientation of lemniscate along x)
lemni_type = 0                                  # 0 = surv, 1 = rolling, 2 = mobbing
quat_0 = quat.e2q(np.array([0,0,0]))           # if lemniscate, this has to be all zeros (consider expanding later to rotate the whole swarm)
quat_0_ = quat.quatjugate(quat_0)               # used to untwist                               

# range parameters 
d_alpha = 5             # [note: only used for Saber flocking] lattice scale (Saber flocking, distance between a-agents)
r_alpha = 2*d_alpha     # range at which neighbours can be sensed (for Saber flocking, this is the interaction range of a-agents)
d_beta = 2              # [note: only used for Saber flocking] desired separation (Saber flocking, distance between a- and b-agents)
r_beta = 2*d_beta       # range at which obstacles can be sensed, (for Saber flocking, this is the interaction range of a- and b-agents)


# legacy
# ------
#d = 5                       # lattice scale (Saber flocking, distance between a-agents)
#r = 2*d                     # range at which neighbours can be sensed (Saber flocking, interaction range of a-agents)
#d_prime = 2 #0.6*d          # desired separation (Saber flocking, distance between a- and b-agents)
#r_prime = 2*d_prime         # range at which obstacles can be sensed, (Saber flocking, interaction range of a- and b-agents)


# Vehicles states
# ---------------
state = np.zeros((6,nVeh))
state[0,:] = iSpread*(np.random.rand(1,nVeh)-0.5)                   # position (x)
state[1,:] = iSpread*(np.random.rand(1,nVeh)-0.5)                   # position (y)
state[2,:] = np.maximum((iSpread*np.random.rand(1,nVeh)-0.5),2)+5  # position (z)
state[3,:] = 0                                                  # velocity (vx)
state[4,:] = 0                                                  # velocity (vy)
state[5,:] = 0                                                  # velocity (vz)
centroid = encircle_tools.centroid(state[0:3,:].transpose())

# Commands
# --------
cmd = np.zeros((3,nVeh))
cmd[0] = np.random.rand(1,nVeh)-0.5      # command (x)
cmd[1] = np.random.rand(1,nVeh)-0.5      # command (y)
cmd[2] = np.random.rand(1,nVeh)-0.5      # command (z)

# Targets
# -------
targets = 4*(np.random.rand(6,nVeh)-0.5)
targets[0,:] = -1 #5*(np.random.rand(1,nVeh)-0.5)
targets[1,:] = -1 #5*(np.random.rand(1,nVeh)-0.5)
targets[2,:] = 15
targets[3,:] = 0
targets[4,:] = 0
targets[5,:] = 0
targets_encircle = targets.copy()
error = state[0:3,:] - targets[0:3,:]

#%% Define obstacles (kind of a manual process right now)
# ------------------------------------------------------
nObs    = 0     # number of obstacles 
vehObs  = 0     # include other vehicles as obstacles [0 = no, 1 = yes] 


# if escorting, need to generate an obstacle 
if nObs == 0 and escort == 1:
    nObs = 1

obstacles = np.zeros((4,nObs))
oSpread = iSpread

# manual (comment out if random)
# obstacles[0,:] = 0    # position (x)
# obstacles[1,:] = 0    # position (y)
# obstacles[2,:] = 0    # position (z)
# obstacles[3,:] = 0

#random (comment this out if manual)
if nObs != 0:
    obstacles[0,:] = oSpread*(np.random.rand(1,nObs)-0.5)-1                   # position (x)
    obstacles[1,:] = oSpread*(np.random.rand(1,nObs)-0.5)-1                   # position (y)
    obstacles[2,:] = oSpread*(np.random.rand(1,nObs)-0.5)+15                  # position (z)
    #obstacles[2,:] = np.maximum(oSpread*(np.random.rand(1,nObs)-0.5),14)     # position (z)
    obstacles[3,:] = np.random.rand(1,nObs)+2                             # radii of obstacle(s)

# manual - make the target an obstacle
if escort == 1:
    obstacles[0,0] = targets[0,0]     # position (x)
    obstacles[1,0] = targets[1,0]     # position (y)
    obstacles[2,0] = targets[2,0]     # position (z)
    obstacles[3,0] = 0.5              # radii of obstacle(s)

# Walls/Floors 
# - these are defined manually as planes
# --------------------------------------   
nWalls = 1
walls = np.zeros((6,nWalls)) 
walls_plots = np.zeros((4,nWalls))

# add the ground at z = 0:
newWall0, newWall_plots0 = tools.buildWall('horizontal', -2) 

# load the ground into constraints   
walls[:,0] = newWall0[:,0]
walls_plots[:,0] = newWall_plots0[:,0]

# add other planes (comment out by default)

# newWall1, newWall_plots1 = flock_tools.buildWall('diagonal1a', 3) 
# newWall2, newWall_plots2 = flock_tools.buildWall('diagonal1b', -3) 
# newWall3, newWall_plots3 = flock_tools.buildWall('diagonal2a', -3) 
# newWall4, newWall_plots4 = flock_tools.buildWall('diagonal2b', 3)

# load other planes (comment out by default)

# walls[:,1] = newWall1[:,0]
# walls_plots[:,1] = newWall_plots1[:,0]
# walls[:,2] = newWall2[:,0]
# walls_plots[:,2] = newWall_plots2[:,0]
# walls[:,3] = newWall3[:,0]
# walls_plots[:,3] = newWall_plots3[:,0]
# walls[:,4] = newWall4[:,0]
# walls_plots[:,4] = newWall_plots4[:,0]

#%% Run Simulation
# ----------------------
t = Ti
i = 1
f = 0         # parameter for future use

nSteps = int(Tf/Ts+1)
t_all          = np.zeros(nSteps)
states_all     = np.zeros([nSteps, len(state), nVeh])
cmds_all       = np.zeros([nSteps, len(cmd), nVeh])
targets_all    = np.zeros([nSteps, len(targets), nVeh])
obstacles_all  = np.zeros([nSteps, len(obstacles), nObs])
centroid_all   = np.zeros([nSteps, len(centroid), 1])
f_all          = np.ones(nSteps)
lemni_all      = np.zeros([nSteps, nVeh])

# initial metrics
#metrics_order_all = np.zeros(nSteps)
#metrics_order = 0
metrics_order_all = np.zeros((nSteps,5))
metrics_order = np.zeros((1,5))

t_all[0]                = Ti
states_all[0,:,:]       = state
cmds_all[0,:,:]         = cmd
targets_all[0,:,:]      = targets
obstacles_all[0,:,:]    = obstacles
centroid_all[0,:,:]     = centroid
f_all[0]                = f
metrics_order_all[0,:]    = metrics_order

lemni = np.zeros([1, nVeh])
lemni_all[0,:] = lemni

if tactic_type == 'lemni':
    twist_perp = lemni_tools.enforce(ref_plane, tactic_type, quat_0)
elif tactic_type == 'statics':
    twist_perp = statics.enforce(ref_plane, tactic_type, quat_0)
    

#%% start the simulation
# --------------------

while round(t,3) < Tf:
  

    # if mobbing, offset targets back down
    if tactic_type == 'lemni' and lemni_type == 2:
        targets[2,:] -= r_desired
    
    # Evolve the target
    # -----------------
    #tSpeed = 0
    targets[0,:] = targets[0,:] + tSpeed*0.002
    targets[1,:] = targets[1,:] + tSpeed*0.005
    targets[2,:] = targets[2,:] + tSpeed*0.0005
    
    # Update the obstacle
    # manual - make the target an obstacle
    if escort == 1:
        obstacles[0,0] = targets[0,0]     # position (x)
        obstacles[1,0] = targets[1,0]     # position (y)
        obstacles[2,0] = targets[2,0]     # position (z)

    # Evolve the states
    # -----------------
    state = node.evolve(Ts, state, cmd)
    #state = node.evolve_sat(Ts, state, cmd)
    
    # Store results
    # -------------
    t_all[i]                = t
    states_all[i,:,:]       = state
    cmds_all[i,:,:]         = cmd
    targets_all[i,:,:]      = targets
    obstacles_all[i,:,:]    = obstacles
    centroid_all[i,:,:]     = centroid
    f_all[i]                = f
    lemni_all[i,:]          = lemni
    metrics_order_all[i,:]    = metrics_order
    
    # Increment 
    # ---------
    t += Ts
    i += 1
        
    #%% Compute Trajectory
    # --------------------
         
    #if flocking
    if tactic_type == 'reynolds' or tactic_type == 'saber' :
        trajectory = targets 
    
    # if encircling
    if tactic_type == 'circle': 
        # compute trajectory
        trajectory, _ = encircle_tools.encircle_target(targets, state, r_desired, phi_dot_d, ref_plane, quat_0)
    
    # if lemniscating
    elif tactic_type == 'lemni':
        # compute trajectory
        trajectory, lemni = lemni_tools.lemni_target(nVeh,r_desired,lemni_type,lemni_all,state,targets,i,unit_lem,phi_dot_d,ref_plane,quat_0,t,twist_perp)
    
    # if static shapes  
    elif tactic_type == 'statics':
        # compute trajectory
        trajectory, lemni = statics.lemni_target(nVeh,r_desired,lemni_type,lemni_all,state,targets,i,unit_lem,phi_dot_d,ref_plane,quat_0,t,twist_perp)
 



             
    #%% Prep for compute commands (next step)
    # ----------------------------
    states_q = state[0:3,:]     # positions
    states_p = state[3:6,:]     # velocities 
    
    # Compute metrics
    # ---------------
    centroid = tools.centroid(state[0:3,:].transpose())
    swarm_prox = tools.sigma_norm(centroid.ravel()-targets[0:3,0])
    metrics_order[0,0] = swarm_metrics.order(states_p)
    metrics_order[0,1:5] = swarm_metrics.separation(states_q,targets[0:3,:],obstacles)
    
    # Add other vehicles as obstacles (optional, default = 0)
    # -------------------------------------------------------
    #vehObs = 0     # include other vehicles as obstacles [0 = no, 1 = yes]  
    if vehObs == 0: 
        obstacles_plus = obstacles
    elif vehObs == 1:
        #states_plus = np.vstack((state[0:3,:], d_prime*np.ones((1,state.shape[1])))) 
        states_plus = np.vstack((state[0:3,:], d_beta*np.ones((1,state.shape[1])))) 
        obstacles_plus = np.hstack((obstacles, states_plus))
            
    #%% Compute the commads (next step)
    # --------------------------------       
    #cmd = tactic.commands(states_q, states_p, obstacles_plus, walls, r, d, r_prime, d_prime, targets[0:3,:], targets[3:6,:], trajectory[0:3,:], trajectory[3:6,:], swarm_prox, tactic_type, centroid, escort)
    cmd = tactic.commands(states_q, states_p, obstacles_plus, walls, r_alpha, d_alpha, r_beta, d_beta, targets[0:3,:], targets[3:6,:], trajectory[0:3,:], trajectory[3:6,:], swarm_prox, tactic_type, centroid, escort)
      
    
#%% Produce animation of simulation
# ---------------------------------
showObs = 1 # (0 = don't show obstacles, 1 = show obstacles, 2 = show obstacles + floors/walls)
#ani = animation.animateMe(Ts, t_all, states_all, cmds_all, targets_all[:,0:3,:], obstacles_all, d, d_prime, walls_plots, showObs, centroid_all, f_all, r_desired, tactic_type)
ani = animation.animateMe(Ts, t_all, states_all, cmds_all, targets_all[:,0:3,:], obstacles_all, d_alpha, d_beta, walls_plots, showObs, centroid_all, f_all, r_desired, tactic_type)
#p
#plt.show()    

#%% Save stuff

pickle_out = open("Data/t_all.pickle","wb")
pickle.dump(t_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/cmds_all.pickle","wb")
pickle.dump(cmds_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/states_all.pickle","wb")
pickle.dump(states_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/targets_all.pickle","wb")
pickle.dump(targets_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/obstacles_all.pickle","wb")
pickle.dump(obstacles_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/centroid_all.pickle","wb")
pickle.dump(centroid_all, pickle_out)
pickle_out = open("Data/lemni_all.pickle","wb")
pickle.dump(lemni_all, pickle_out)
pickle_out.close()

