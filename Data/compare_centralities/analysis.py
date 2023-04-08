#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:08:23 2022

@author: tjards
"""

#%% import stuff
# --------------
import numpy as np
import matplotlib.pyplot as plt
import pickle


#%% functions
# ---------

def collect(data):
    
    #data = np.abs(data)
    data2 = np.zeros((data.shape[0],data.shape[2])) 
    
    
    #data_out = np.zeros((data.shape[0]))
    means = np.zeros((data.shape[0])) #np.mean(g_cmds) 
    maxes = np.zeros((data.shape[0])) #np.var(seps_obs)
    mines = np.zeros((data.shape[0])) #np.var(seps_obs)
    
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[2]):
            for k in range(0,data.shape[1]):
                data2[i,j] += data[i,k,j]**2
            data2[i,j] = np.divide(np.sqrt(data2[i,j]),data.shape[1])
            
            # cumulative? (comment out if not)
            data2[i,j] += data2[i-1,j]

        
        #data_out[i] = np.divide(np.sqrt(data_out[i]),data.shape[1]+data.shape[2])
        #data_out[i] = np.divide((data_out[i]),data.shape[1]+data.shape[2])
        #means[i]    = np.mean(data[i,:,:])
        #maxes[i]    = np.maximum(data[i,:,:])
        #mines[i]    = np.minimum(data[i,:,:])
        
        # this works 
        means[i]    = np.mean(data2[i,:])
        maxes[i]    = np.amax(data2[i,:])
        mines[i]    = np.amin(data2[i,:])
        
    return means, maxes, mines
        


#%% data
# -----
with open('04_gramian.pickle', 'rb') as file:
    g_cmds_i = pickle.load(file)

with open('04_degree.pickle', 'rb') as file:
    d_cmds_i = pickle.load(file)
    
with open('04_between.pickle', 'rb') as file:
    b_cmds_i = pickle.load(file)

with open('04_between_min.pickle', 'rb') as file:
    bm_cmds_i = pickle.load(file)



g_cmds, g_maxes, g_mines = collect(g_cmds_i)
d_cmds, d_maxes, d_mines = collect(d_cmds_i)
b_cmds, b_maxes, b_mines = collect(b_cmds_i)
bm_cmds, bm_maxes, bm_mines = collect(bm_cmds_i)
    
with open('04_gt.pickle', 'rb') as file:
    g_t = pickle.load(file)


#%% plots
# --------

s   =   0
e   =   2500


fig, ax = plt.subplots()
ax.plot(g_t[s:e],g_cmds[s:e],'-b')
#ax.plot(g_t[:],g_maxes[:],':b')
#ax.plot(g_t[:],g_mines[:],':b')
#ax.fill_between(g_t[:], g_maxes[:], g_mines[:], color = 'blue', alpha = 0.1)


ax.plot(g_t[s:e],d_cmds[s:e],'-g')
#ax.plot(g_t[:],d_maxes[:],':g')
#ax.plot(g_t[:],d_mines[:],':g')
#ax.fill_between(g_t[:], d_maxes[:], d_mines[:], color = 'green', alpha = 0.1)

ax.plot(g_t[s:e],b_cmds[s:e],'-m')
#ax.plot(g_t[:],b_maxes[:],':k')
#ax.plot(g_t[:],b_mines[:],':k')
#ax.fill_between(g_t[:], b_maxes[:], b_mines[:], color = 'black', alpha = 0.1)

ax.plot(g_t[s:e],bm_cmds[s:e],'-k')
#ax.plot(g_t[:],bm_maxes[:],':k')
#ax.plot(g_t[:],bm_mines[:],':k')
#ax.fill_between(g_t[:], bm_maxes[:], bm_mines[:], color = 'black', alpha = 0.1)


#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', ylabel='Total Energy Consumption [ms^2]',
       title='Comparison on Pin Selection for Assembly (20 agents)')
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
ax.grid()
#ax.legend(['gramian', 'degree', 'between'])
ax.legend(['Controllability Gramian', 'Degree Centrality','Max Betweenness', 'Min Betweenness'])
#ax.set(xlim=(0, g_t[e]), ylim=(0, 1))


#fig.savefig("test.png")
plt.show()