from grid import *
import operator
import tkinter as tk
import numpy as np
import random
import matplotlib.pyplot as plt
 
class Action:
    type = typed_property('type', str)
    movement = typed_property('movememt', (int, int))
    prob = typed_property('prob', float)

    def __init__(self, type = None, movement = None, prob = None) -> None:
        self.type = type            # type of action (up 'u', down 'd',...)
        self.movement = movement    # how it effects location of next block
        self.prob = prob            # probability of choosing this action

def sys_dyn(map_, s, a):
    s_next = Block()
    s_next.loc = tuple(map(operator.add, s.loc ,a.movement))
    s_next.type = 'free'
    r = -1

    # going into the terminal state
    if s.type == 'trap':
        s_next.loc = map_.start
        s_next.type = 'start'
        r = 0
    elif s.type == 'goal':
        s_next.loc = map_.terminal
        s_next.type = 'terminal'
        r = 0
    # hitting the walls
    elif s_next.loc[0] < 0 or s_next.loc[0] >= map_.height or\
     s_next.loc[1] < 0 or s_next.loc[1] >= map_.width:
        s_next.loc = s.loc      # bounces back
    # hitting the block
    elif s_next.loc == map_.blocked:
        s_next.loc = s.loc      # bounces back
    # into the trap
    elif s_next.loc in map_.trap:
        s_next.type = 'trap'
        r = -100
    # reaching the goal
    elif s_next.loc == map_.goal:
        s_next.type = 'goal'
        r = 10
    
    for block in map_.grid:
        if block.loc == s_next.loc:
            s_next.Qvalue = block.Qvalue

    return s_next, r

#def policy_eval(map_, g, policy, delta):
    
    # theta and Delta values
    theta = 0.00001; Delta = 100
    iter = 0

    #init value randomly for each state
    for state in g:
        state.value = 0

    while Delta > theta:
        Delta = 0
        iter += 1

        # iterate over all states
        for block in g:

            if block.loc == map_.blocked:
                continue
            elif block.loc == map_.trap:
                r_step = -10
            elif block.loc == map_.goal:
                r_step = 10
            else:
                r_step = -1

            v = block.value
            temp = 0
            for act in policy:
                s_next = sys_dyn(map_, block, act)
                temp = temp + act.prob*(r_step + delta*s_next.value)
            block.value = temp
            Delta = max(Delta, abs(v - block.value))
        #showgrid(g)
    return g

def action(eps, s, policy):

    greed_ = np.random.choice([0, 1], 1, p = [eps, 1-eps])

    if greed_ == 0:
        act = random.choice(policy)
    else:
        m = max(s.Qvalue.values())
        #print(m, end=' ')
        keylist = []
        for key, value in s.Qvalue.items():
            if value == m:
                keylist.append(key)
        #print(keylist)
        a = random.choice(keylist)
        for action in policy:
            if a == action.type:
                act = action
    
    return act

def SARSA(map_, policy, alpha, eps, delta):
    superiter = 0; rwd = [0]*500; rwds = []
    while superiter < 50:
        
        #init Q value for each block
        for block in map_.grid:
            block.Qvalue = {}
            for act in policy:
                block.Qvalue[act.type] = 0
        
        iter = 0; 
        while iter < 500:
            for block in map_.grid:
                if block.loc == map_.start:
                    st = block

            act = action(eps, st, policy)
            rd = 0  # init reward per episode
            while st.type != 'terminal':
                #print(st.loc, end=' '), print(st.type, end=' '); print(act.type, end=' '); print(st.Qvalue)
                s_next, r = sys_dyn(map_, st, act)
                
                #print(s_next.loc, end=' '), print(s_next.type, end=' '), print(act.type)
                actp = action(eps, s_next, policy)
                #print(s_next.loc, end=' '); print(actp.type, end=' ')

                st.Qvalue[act.type] = st.Qvalue[act.type] + \
                    alpha*(r + delta*s_next.Qvalue[actp.type] - st.Qvalue[act.type])
                st = s_next
                act = actp
                rd += r     # total rewards in episode

            rwd[iter] += rd
            if superiter == 49:
                rwds.append(rd)
            iter = iter + 1
        superiter += 1
        print(superiter)

    rwd = [numb/50 for numb in rwd]
 
    #i = 0; window_size = 10
    #moving_averages = []
    #while i < len(rwds) - window_size + 1:
    #    this_window = rwds[i : i + window_size]
    #    window_average = sum(this_window)
    #    moving_averages.append(window_average)
    #    i += 1
    
    plt.figure(0)
    plt.plot(rwd); 
    plt.draw()
    #plt.plot(rwd)
    #plt.draw()

    #showgrid(map_.grid)

    return map_

def Q_learning(map_, policy, alpha, eps, delta):
    superiter = 0; rwd = [0]*500; rwds = []
    while superiter < 50:
    
    #init Q value for each block
        for block in map_.grid:
            block.Qvalue = {}
            for act in policy:
                block.Qvalue[act.type] = 0
        
        iter = 0; 
        while iter < 500:
            for block in map_.grid:
                if block.loc == map_.start:
                    st = block

            act = action(eps, st, policy)
            rd = 0
            while st.type != 'terminal':
                #print(st.loc, end=' '), print(st.type, end=' '); print(act.type, end=' '); print(st.Qvalue)
                s_next, r = sys_dyn(map_, st, act)
                
                #print(s_next.loc, end=' '), print(s_next.type, end=' '), print(act.type)
                actp = action(eps, s_next, policy)
                bestAct = action(1, s_next, policy)
                #print(s_next.loc, end=' '); print(actp.type, end=' ')

                st.Qvalue[act.type] = st.Qvalue[act.type] + \
                    alpha*(r + delta*s_next.Qvalue[bestAct.type] - st.Qvalue[act.type])
                st = s_next
                act = actp
                rd += r
            
            rwd[iter] += rd
            if superiter == 49:
                rwds.append(rd)
            iter = iter + 1
        superiter += 1
        print(superiter)

    rwd = [numb/50 for numb in rwd]

    plt.figure(0)
    plt.plot(rwd)
    plt.draw()
    plt.show()

    showgrid(map_.grid)

    return map_

def genRandPolicy():
    types = ['u', 'd', 'r', 'l']
    a = []

    for type in types:
        if type == 'u':
            act = Action(type, (-1,0), 0.25)
        elif type == 'd':
            act = Action(type, (1,0), 0.25)
        elif type == 'r':
            act = Action(type, (0,1), 0.25)
        elif type == 'l':
            act = Action(type, (0,-1), 0.25)
        a.append(act)
    return a

def showgrid(grid):
    master = tk.Tk()

    # Create the label objects and pack them using grid
    for block in grid:
        if block.loc == (-1,-1):
            pass
        else:
            m = max(block.Qvalue.values())
            for key, value in block.Qvalue.items():
                if value == m:
                    bestAction = key
            tk.Label(master, text = str(bestAction)). grid(row=block.loc[0], column=block.loc[1])
    
    # The mainloop
    tk.mainloop()

def main():
    g = Grid(); g.createGrid()
    g1 = Grid(); g1.createGrid()
    delta = 1
    policy = genRandPolicy()
    alpha = 0.2 # step-size 
    eps = 0.1   # probability of exploration
    delta = 1
    #grid = policy_eval(g, grid, policy, delta)
    g_SARSA = SARSA(g, policy,alpha, eps, delta)
    g_QLearn = Q_learning(g1, policy,alpha, eps, delta)
    #showgrid(grid)
    #print(policy[2].movement)
   

    #for block in grid:
    #    if block.loc == (3,1):
    #        n = sys_dyn(g, block, policy[3])
    #        print(n.loc, end=' '), print(n.type)
    #    print(block.loc, end=' '), print(block.type)

if __name__ == '__main__':
    main()