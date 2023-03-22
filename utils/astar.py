#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:21:07 2023

@author: tjards
"""

# betweenness centrality: expresses the significance of a node to network connectivity
#   count the number of times a node appears on the shortest path between any two other vertices


# import stuff
# ------------

import numpy as np
import random
from collections import defaultdict
import heapq 

# parameters
# ----------
r       = 5
nNodes  = 10


# # random adj matrix
# # -----------------
# A = np.ones((10,10))

# for i in range(0,10):
    
#     A[random.randint(0, A.shape[0]-1),random.randint(0, A.shape[0]-1)] = 0
#     A[random.randint(0, A.shape[0]-1),random.randint(0, A.shape[0]-1)] = 0

# for i in range(0,A.shape[0]):
#     A[i,i] = 0
    

data = 10*np.random.rand(3,nNodes)

#%% build Graph (as dictionary)
# ----------------------------
def build_graph(data):
    G = {}
    nNodes  = data.shape[1]     # number of agents (nodes)
    # for each node
    for i in range(0,nNodes):
        # create a set of edges
        set_i = set()
        # search through neighbours (will add itself)
        for j in range(0,nNodes):
            # compute distance
            dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
            # if close enough
            if dist < r:
                # add to set_i
                set_i.add(j)
            #else:
            #    print("debug: ", i," is ", dist, "from ", j)
        G[i] = set_i
    return G

G = build_graph(data)

# define djikstra (shortest path)
# -------------------------------

# accepts a Graph (dictionary) and starting node
def search_djikstra(G, source):
    
    closed = set()                              # set of nodes not to visit (or already visited)
    parents = {}                                # stores the path back to source 
    costs = defaultdict(lambda: float('inf'))   # store the cost, with a default value of inf for unexplored nodes
    costs[source] = 0
    que = []                                    # to store cost to the node from the source
    heapq.heappush(que,(costs[source],source))  # push element into heap in form (cost, node)
    # note: heap is a binary tree where parents hold smaller values than their children
    
    # while there are elements in the heap
    while que:
        
        # "i" is the index for the node being explored in here
        cost_i, i = heapq.heappop(que)          # returns smallest element in heap, then removes it
        closed.add(i)                           # add this node to the closed set
        
        # search through neighbours
        for neighbour in G[i]:
            
            # we'll define each hop with a weight of 1, but this could be distance (later)
            w = 1
            
            # don't explore nodes in closed set
            if neighbour in closed:
                continue
            
            # update cost
            cost_update = costs[i] + w
            
            # if updated cost is less than current (default to inf, hence defaultdict)
            if  cost_update < costs[neighbour]:
                
                #store the and parents and costs
                parents[neighbour] = i
                costs[neighbour] = cost_update
                # add to heap
                heapq.heappush(que, (cost_update, neighbour))

    return parents, costs
    
parents, costs = search_djikstra(G, 0)
