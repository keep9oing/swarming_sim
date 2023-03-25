#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:21:07 2023

@author: tjards
"""

# betweenness centrality: expresses the significance of a node to network connectivity
#   count the number of times a node appears on the shortest path between any two other vertices


# Import stuff
# ------------
import numpy as np
import random
from collections import defaultdict, Counter
import heapq 

# Parameters
# ----------
r       = 5
nNodes  = 10
data = 10*np.random.rand(3,nNodes)

#%% Build Graph (as dictionary)
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

#%% Djikstra's shortest path
# --------------------------
# accepts a Graph and starting node
# finds the shortest path from source to all other nodes

def search_djikstra(G, source):
    
    closed = set()                              # set of nodes not to visit (or already visited)
    parents = {}                                # stores the path back to source 
    costs = defaultdict(lambda: float('inf'))   # store the cost, with a default value of inf for unexplored nodes
    costs[source] = 0
    queue = []                                    # to store cost to the node from the source
    heapq.heappush(queue,(costs[source],source))  # push element into heap in form (cost, node)
    # note: heap is a binary tree where parents hold smaller values than their children
    
    # while there are elements in the heap
    while queue:
        
        # "i" is the index for the node being explored in here
        cost_i, i = heapq.heappop(queue)        # returns smallest element in heap, then removes it
        closed.add(i)                           # add this node to the closed set
        
        # search through neighbours
        for neighbour in G[i]:
            
            # we'll define each hop/step with a cost of 1, but this could be distance (later)
            step_cost = 1
            
            # don't explore nodes in closed set
            if neighbour in closed:
                continue
            
            # update cost
            cost_update = costs[i] + step_cost
            
            # if updated cost is less than current (default to inf, hence defaultdict)
            if  cost_update < costs[neighbour]:
                
                #store the and parents and costs
                parents[neighbour] = i
                costs[neighbour] = cost_update
                
                # add to heap
                heapq.heappush(queue, (cost_update, neighbour))

    return parents, costs
    


#%% Adjacency Matrix
# ------------------------------

# A = {a_ij} s.t. 1 if i,j are neighbours, 0 if not
def adj_matrix(data):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    A       = np.zeros((nNodes,nNodes)) # initialize adjacency matrix as zeros
    # for each node
    for i in range(0,nNodes):  
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    A[i,j] = 1
    # ensure A = A^T
    assert (A == A.transpose()).all()
    # return the matrix
    return A


#%% Compute the Degree Matrix
# ------------------------------
# D = diag{d1,d2,...dN}
def deg_matrix(data):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    D       = np.zeros((nNodes,nNodes)) # initialize degree matrix as zeros
    # for each node
    for i in range(0,nNodes):
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    D[i,i] += 1
    # return the matrix
    return D

#%% Compute the graph Laplacian
# -----------------------------
def lap_matrix(A,D):
    L = D-A
    eigs = np.linalg.eigvals(L)         # eigen values 
    # ensure L = L^T
    assert (L == L.transpose()).all()
    # ensure has zero row sum
    assert L.sum() == 0
    # ensure Positive Semi-Definite (all eigen values are >= 0)
    assert (eigs >= 0).all()
    # return the matrix
    return L



#%% compute Betweenness Centrality
# ------------------------------
def betweenness(G):
    
    # store the betweenness for each node
    all_paths = {}
    k = 0
    
    # create a nested dict of all the shortest paths 
    for i in range(0,len(G)):
        
        parents, _      = search_djikstra(G,i)
        all_paths[k]    = parents
        k += 1
    
    # count all the influencers (i.e. those on shortest paths)
    influencers = count_influencers(all_paths)
    # sum of all paths
    summ = len(G)*(1+len(G))/2
    
    return {n: influencers[n] / summ for n in influencers}


#%% count instances of node appearing in a shortest path
# ------------------------------------------------------
def count_influencers(all_paths):

    influencers = defaultdict(lambda: float(0))
    
    # do all this for each destination nodes
    for k in range(0,len(G)):
        
        # create a set of all other nodes
        search = set(range(0,len(G)))
        search.remove(k)
    
        # we will search through all the other nodes
        while search:
        
            # select and remove a node
            i = search.pop()
            
            # this will be part of the search
            sub_search = set()
            sub_search.add(i)
            
            # but so will others we find along the way (sub search)
            while sub_search:
                
                j = sub_search.pop()
            
                # identify the parent 
                parent_i = all_paths[k][j]
                
                # if this this the destination
                if parent_i == k:
                    # get out
                    continue
                # else, keep looking
                else:
                
                    # add this to the subsearch
                    sub_search.add(parent_i)
            
                    # count this parent as part of a path
                    influencers[parent_i] += 1
                    #print(parent_i, 'is a parent of', j, 'of ...', k )
        
    return influencers 


#%% testing
# --------

G               = build_graph(data)
parents, costs  = search_djikstra(G, 0)
adjacency       = adj_matrix(data)
degree          = deg_matrix(data)
laplacian       = lap_matrix(adjacency, degree)
betweennesses   = betweenness(G)


    
    

    
    
    
    



