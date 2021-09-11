#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This program computes some useful swarm metrics

@author: tjards

"""

import numpy as np


#%% Order
# -------

def order(states_p):

    order = 0
    N = states_p.shape[1]
    # if more than 1 agent
    if N > 1:
        # for each vehicle/node in the network
        for k_node in range(states_p.shape[1]):
            # inspect each neighbour
            for k_neigh in range(states_p.shape[1]):
                # except for itself
                if k_node != k_neigh:
                    # and start summing the order quantity
                    norm_i = np.linalg.norm(states_p[:,k_node])
                    if norm_i != 0:
                        order += np.divide(np.dot(states_p[:,k_node],states_p[:,k_neigh]),norm_i**2)
            # average
            order = np.divide(order,N*(N-1))
            
    return order