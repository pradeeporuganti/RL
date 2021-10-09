import matplotlib.pyplot as plt
import random
import time; from statistics import mean
import os

def optimbellman(s, V, p_h):
    Q = []; A = [_ for _ in range(0, min(s, 100-s)+1)]
    for a in A:
        Q.append(p_h*(V[s+a]) + (1-p_h)*(V[s-a]))
    return max(Q)
    
def value_iter(S, V, p_h):
    _Delta = 1
    _theta = 1e-8
    iter = 0

    # get values
    while _Delta > _theta:
        _Delta = 0
        for s in S:
            v = V[s]                               
            val = optimbellman(s, V, p_h)        
            V[s] = val
            _Delta = max(_Delta, abs(v - val))
        iter += 1
    #print(iter)

    # get policy
    pi_ = [None]*101
    for s in S:
        A = [_ for _ in range(0, min(s, 100-s)+1)]; Q = []
        for a in A:
            Q.append(p_h*(V[s+a]) + (1-p_h)*(V[s-a]))
        # handling deminal points and multiple max
        if Q.index(max(Q)) == 0:
            bet = [i for i, x in enumerate(Q) if x == max(Q)]
            pi_[s] = A[max(bet)]
        else:   pi_[s] = A[Q.index(max(Q))]
    
    return pi_, V

def gen_xp(policy, S, p_h):
    #S0 = random.choice(S[1:-1])     # randomly choose initial capital
    S0 = 1
    S_xprnc = [S0]; R = [0]

    while True:
        a = policy[S_xprnc[-1]]
        act = random.choices([a, -a], weights=(p_h, 1-p_h), k=1)
        s_next = S_xprnc[-1] + act[0]
        S_xprnc.append(s_next)
        if s_next != 100:
            R.append(0)
        else: R.append(1)
        # terminate episode when reaching $100 or $0 (broke)
        if s_next  == 0 or s_next == 100:
            break

    return S_xprnc, R


def monte_carlo(policy, S, p_h):
    
    Returns = dict.fromkeys(S, [])
    
    estim_V = [0]*len(S)
    iter_ = 0; iter_time = []
    while iter_ < 151:
        start_time = time.time()                # start time to gen xp and get values
        xprnc, rwd = gen_xp(policy, S, p_h)
        G = 0
        for index, value in enumerate(reversed(xprnc)):
            G = G + rwd[len(xprnc) - (index+1)]
            Returns[value].append(G)
            estim_V[value] = mean(Returns[value])
        iter_ += 1
        end_time = time.time()                  # end time
        print(iter_, end=" ")
        print(end_time - start_time, end=" ")
        print(len(xprnc))
        iter_time.append(end_time - start_time)
        if iter_%50 == 0:
            plt.figure(3)
            plt.plot(estim_V, label="iteration" + str(iter_))
            plt.legend()
            plt.draw()
            print(estim_V)
    plt.show()

    return estim_V

def main():
    p_h = 0.55 

    S = [_ for _ in range(0, 101)]
    V = [0]*101; V[-1] = 1

    policy, finalV = value_iter(S, V, p_h)
    print(policy[51])
    estim_V = monte_carlo(policy, S, p_h)
    
    plt.figure(0)
    plt.plot(S, finalV, label = "VI")
    plt.xlabel("Captial")
    plt.ylabel("Value")
    plt.draw()
    
    plt.figure(1)
    plt.plot(S, policy)
    plt.xlabel("Captial"); plt.ylabel("Bet")
    plt.draw()
    plt.show()

    #estim_V = monte_carlo(policy, S)

if __name__ == '__main__':
    main()