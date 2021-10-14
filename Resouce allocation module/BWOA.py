import numpy as np
import random as rand
from random import *
import math
import copy

def WOA(SearchAgents_no, Max_iter, fobj, dim, cases, cur):
    '''
    This code is for Discrete Whale Optimization Algorithm(DWOA)
    "SearchAgents_no" is number of searching agents
    "Max_iter" is limit of iteration
    "fobj" denotes the objective function
    "dim" is number of split num
    "cases" is number of MEC num
    '''
    Leader_pos = np.zeros((1,dim))
    Leader_score = float('-inf')
    Leader_score_pre = Leader_score
    
    delta = 1e-6
    todoTol = 0
    Flag = 0

    Positions = np.zeros((SearchAgents_no, dim))
    Positions[0] = copy.deepcopy(cur)
    for i in range(1, len(Positions)):
        for j in range(len(Positions[i])):
            Positions[i][j] = int(rand.random() * (cases + 1))
    Convergence_curve = np.zeros((1, Max_iter))

    iter = 0

    while iter < Max_iter and Flag <= 3:
        # Leader_score = 0
        for i in range(len(Positions)):
            for j in range(len(Positions[i])):
                if Positions[i][j] >= cases:
                    Positions[i][j] = cases - 1
                elif Positions[i][j] < 0:
                    Positions[i][j] = 0
                Positions[i][j] = math.floor(Positions[i][j])
            fitness = fobj(Positions[i])
            if fitness > Leader_score:
                Leader_score = fitness
                Leader_pos = copy.deepcopy(Positions[i])
                print(fitness)

        a = 2 - iter * ((2)/Max_iter)
        a2 = -1 + iter * ((-1)/Max_iter)

        for i in range(len(Positions)):
            r1 = rand.random()
            r2 = rand.random()

            A = 2 * a * r1 - a
            C = 2 * r2

            p = rand.random()

            b = 1
            l = (a2-1)*rand.random() + 1

            for j in range(len(Positions[i])):
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(SearchAgents_no * rand.random())
                        X_rand = Positions[rand_leader_index]
                        D_X_rand = abs(C*X_rand[j] - Positions[i][j])
                        Positions[i][j] = X_rand[j] - A*D_X_rand
                    elif abs(A) < 1:
                        D_Leader = abs(C*Leader_pos[j] - Positions[i][j])
                        Positions[i][j] = Leader_pos[j] - A*D_Leader
                elif p >= 0.5:
                    distance2Leader = abs(Leader_pos[j]-Positions[i][j])
                    Positions[i][j] = distance2Leader*math.exp(b*l)*math.cos(2*math.pi*l) + Leader_pos[j]

        iter = iter + 1
        Convergence_curve[0][iter-1] = Leader_score
        if todoTol == 1 and abs(Leader_score - Leader_score_pre) < delta:
            Flag = Flag + 1
            # Convergence_curve = Convergence_curve[0][0:iter]
        
        Leader_score_pre = Leader_score
    
    converg.append(Convergence_curve[0])
    return Leader_pos

# Fitness for one agent
def fobj(Position):
    return fit
