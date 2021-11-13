import matplotlib.pyplot as plt
import numpy as np
import math 

class World():
    S_tot = 1000    # total number of states
    S_i = np.arange(1, S_tot+1)     # states to iterate over;
    S_term = [0, 1001]
    # 0 and 1001 are terminal states on left and right
    S_0 = 500       # init 
    S_act = [-1, 1];    # going left (-1) and right (+1)
    step = 100

def optimbellman(w, s, V):     # lifted and modified from HW1
    temp = 0
    for a in w.S_act:
        for st in range(1, w.step+1):
            s_next = s + (a*st)
            s_next =  max(min(s_next, w.S_tot + 1), 0)      # no overshoot beyond edges
            
            if s_next == w.S_tot + 1:
                r = 1
            elif s_next == 0:
                r = -1
            else: r = 0
            temp += (1/200) * (r + V[s_next])   # handling overshoot by adding probability. p = 1/200
            # no argmax as transition into these states has equal probability (no action as such)
    V[s] = temp        
    return V

def V_act(w):       # function to calculate real v(s); lifted and modified from HW1

    _Delta = 1
    _theta = 1e-4
    V_guess =  np.zeros(np.shape(w.S_i))
    V_guess = np.concatenate([[0],V_guess,[0]])
    
    # compute true value 
    while _Delta > _theta:
        _Delta = 0
        for s in w.S_i:
            v = V_guess[s]
            V_guess = optimbellman(w, s, V_guess)
            _Delta = max(_Delta, abs(v - V_guess[s]))
    
    return V_guess

def genXP(w):
    S = []; A = []; R = []
    S.append(w.S_0)

    while S[-1] not in w.S_term:

        a = np.random.choice(w.S_act, 1, p=[0.5, 0.5])      # get side
        step = np.random.choice(np.arange(1, w.step+1), 1)  # get step

        s_next = S[-1] + a.item()*step.item()     # move
        s_next = max(min(s_next, w.S_tot+1), 0) # clip

        A.append(a*step)    # append actions            
        S.append(s_next)    # append states

        if s_next == 1001:
            r = 1
        elif s_next == 0:
            r = -1
        else: r = 0

        R.append(r)         # append rewards 
    
    return S[:-1], A, R

def div_groups(w, s):
    divs = [w.S_i[x:x+100] for x in range(0, len(w.S_i), 100)]

    for i in range(0, len(divs)):
        if s in divs[i]:
            return i            

def state_agg(w):
    num_episodes = 10000
    weights = [0]*10
    alpha = 2e-3
    returns = {k: [] for k in list(w.S_i)}
    
    for ep in range(1, num_episodes+1):
        if ep%100 == 0:
            print(ep)
        S, _, R = genXP(w)
        G = 0

        for ind, s in reversed(list(enumerate(S))):
            G += R[ind]
            returns[s].append(G)
        
        for s in S:
            grp = div_groups(w, s)
            weights[grp] += alpha*((sum(returns[s])/len(returns[s])) - weights[grp])
    
    print(weights)
    return weights, returns

def poly_bases(w):
    num_episodes = 10000
    order = 10
    weights = [0]*(order+1)

    # build polynomial features
    bases = []
    for i in range(0, order+1):
        bases.append(lambda s, i=i:pow(s,i))
    
    alpha = 1e-4
    returns = {k: [] for k in list(w.S_i)}

    for ep in range(1, num_episodes+1):
        if ep%100 == 0:
            print(ep)
        S, _, R = genXP(w)
        G = 0

        for ind, s in reversed(list(enumerate(S))):
            G += R[ind]
            returns[s].append(G)
        
        for s in S:
            s_norm = s/w.S_tot
            feat = [func(s_norm) for func in bases]
            
            for i in range(0, len(weights)):    
                weights[i] += alpha*((sum(returns[s])/len(returns[s]))-np.dot(weights, feat))*feat[i]
    
    print(weights)
    return weights, returns, bases

def four_basis(w):
    num_episodes = 10000
    order = 5
    weights = [0]*(order+1)

    # build polynomial features
    bases = []
    for i in range(0, order+1):
        bases.append(lambda s, i=i:math.cos(math.pi*s*i))

    alpha = 5e-5
    returns = {k: [] for k in list(w.S_i)}

    for ep in range(1, num_episodes+1):
        if ep%100 == 0:
            print(ep)
        S, _, R = genXP(w)
        G = 0

        for ind, s in reversed(list(enumerate(S))):
            G += R[ind]
            returns[s].append(G)
        
        for s in S:
            s_norm = s/w.S_tot
            feat = [func(s_norm) for func in bases]
            
            for i in range(0, len(weights)):    
                weights[i] += alpha*((sum(returns[s])/len(returns[s]))-np.dot(weights, feat))*feat[i]
    
    print(weights)
    return weights, returns, bases
        
if __name__ == '__main__':
    w = World()
    #V_act = V_act(w)
    #weights, returns = state_agg(w); wt = np.zeros(np.size(w.S_i)+1)
    #weights, returns, bases = poly_bases(w)
    weights, returns, bases = four_basis(w)
    v_estim = []; v_estim.append(0) 
    
    for s in w.S_i:
        v_estim.append(0) 
        s_norm = s/1000
        feat = [func(s_norm) for func in bases]
        v_estim[s] = np.dot(weights, feat)

    plt.figure(0)
    plt.plot(v_estim)
    plt.show()


    #print(returns)
    #for s in w.S_i:
    #    grp = div_groups(w, s)
    #    wt[s] = weights[grp]

    #plt.figure(0)
    #plt.plot(list(w.S_i), list(V_act))
    #plt.draw()
    #plt.plot(list(w.S_i), list(wt[1:]))
    #plt.show()


