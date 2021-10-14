import numpy as np
import random as rand
from random import *
import math
from matplotlib import pyplot as plt
import copy
import time
import pandas as pd

def WOA(SearchAgents_no, Max_iter, fobj, dim, lb, ub):
    '''
    This code is for Whale Optimization Algorithm(WOA) to solve minimization problem.
    "SearchAgents_no" is number of searching agents
    "Max_iter" is limit of iteration
    "fobj" denotes the objective function
    "dim" is number of split num
    "lb" is lower bound
    "ub" is upper bound
    '''
    # initialize position vector and score for the leader
    Leader_pos = np.zeros((1,dim))
    # change this to -inf for maximization problems
    Leader_score = float('inf')
    Leader_score_pre = Leader_score
    # tolerance to stop the algorithm
    delta = 1e-6
    Flag = 0
    # Initialize the positions of search agents. Size is SearchAgents_no x dim
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(0, len(Positions)):
        for j in range(len(Positions[i])):
            Positions[i][j] = rand.random() * ub
    Convergence_curve = np.zeros((1, Max_iter))
    # Loop counter
    iter = 0

    ''' Main loop '''
    while iter < Max_iter and Flag <= 3:
        for i in range(len(Positions)):
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(len(Positions[i])):
                if Positions[i][j] > ub:
                    Positions[i][j] = ub
                elif Positions[i][j] < lb:
                    Positions[i][j] = lb
            # Calculate objective function for each search agent        
            fitness = fobj(Positions[i])
            # Update the leader
            if fitness < Leader_score: # Change this to > for maximization problem
                Leader_score = fitness
                Leader_pos = copy.deepcopy(Positions[i])

        # a decreases linearly fron 2 to 0
        a = 2 - iter * ((2)/Max_iter)
        # a2 linearly dicreases from -1 to -2 to calculate t
        a2 = -1 + iter * ((-1)/Max_iter)

        # Update the Position of search agents 
        for i in range(len(Positions)):
            r1 = rand.random()
            r2 = rand.random()

            A = 2 * a * r1 - a
            C = 2 * r2

            b = 1
            l = (a2-1)*rand.random() + 1

            p = rand.random()

            for j in range(len(Positions[i])):
                # follow the shrinking encircling mechanism or prey search
                if p < 0.5:
                    # search for prey (exploration phase)
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(SearchAgents_no * rand.random())
                        X_rand = Positions[rand_leader_index]
                        D_X_rand = abs(C*X_rand[j] - Positions[i][j])
                        Positions[i][j] = X_rand[j] - A*D_X_rand
                    # Shrinking encircling mechanism (exploitation phase)   
                    elif abs(A) < 1:
                        D_Leader = abs(C*Leader_pos[j] - Positions[i][j])
                        Positions[i][j] = Leader_pos[j] - A*D_Leader
                # follow the spiral-shaped path
                elif p >= 0.5:
                    distance2Leader = abs(Leader_pos[j]-Positions[i][j])
                    Positions[i][j] = distance2Leader*math.exp(b*l)*math.cos(2*math.pi*l) + Leader_pos[j]
        # increase the iteration index by 1
        iter = iter + 1
        # negate the objective value (minimization --> maximization problem)
        Convergence_curve[0][iter-1] = -Leader_score

        # check to see whether the stopping criterion is satisifed
        if abs(Leader_score - Leader_score_pre) < delta:
            Flag = Flag + 1
            Convergence_curve = Convergence_curve[0][0:iter]
            Leader_score *= -1
        else:
            Leader_score_pre = Leader_score
    
    return Leader_pos

# Fitness
def fobj(Position):
    fit = 0
    return fit