import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.core.fromnumeric import argmax

class Bandit:

    def __init__(self, numBandits):
        self.numBandits = numBandits
        self.genRewards() 

    def genRewards(self):   # Generates random mean for rewards for each bandit
        self.mu = []
        i = 0
        while i < self.numBandits:
            self.mu.append(np.random.normal())  # True means
            i += 1
        return self.mu

    def getMeans(self, REstim):
        muEstim = np.zeros((self.numBandits, 1))
        for row in range(0, len(REstim)):
            temp = [x for x in REstim[row] if x != 0]
            muEstim[row] = np.mean(temp)
        return muEstim

    def regret(self, R, T):
        muStar = max(self.mu)
        return T*muStar-sum(R)      # best realization of reward I can get over T pulls - what I actually got

    def nGreedy(self, N, iters):    # Explore the first N times and then pick greedy action
        self.N = N
        self.iters = iters
        Q = [0]*self.numBandits
        n = [0]*self.numBandits
        REstim = np.zeros((self.numBandits, iters)) # to estimate the mean of reward distributions from pulls
        R = []                      # to calculate the mean of rewards observed 
        R_t = []
        rollingReg = []

        for i in range(iters):
            if i < N:       # EXPLORATION PHASE
                a = np.random.randint(1, self.numBandits+1)   # randomly choose a bandit to pull
                r = np.random.normal(self.mu[a-1], 1)       # get a random reward with unknown true mean mu and variance 1
                n[a-1] += 1                                   # increase the count of number of times action a was chosen by 1 
                REstim[a-1][i] = r                            # append reward to the sequence of rewards observed
                R.append(r)
                R_t.append(np.mean(R))                      # calculate and append the current mean of rewards to list
                Q[a-1] = Q[a-1] + ((r - Q[a-1])/n[a-1])     # update Q action value
                rollingReg.append(self.regret(R, i))
            else:           # ACT GREEDILY
                a = np.argmax(Q)                            # choose the action with the highest action value
                r = np.random.normal(self.mu[a], 1)         # observe a random reward
                n[a] += 1 
                REstim[a][i] = r                            # append reward to the sequence of rewards observed
                Q[a] = Q[a] + ((r- Q[a])/n[a])              # update Q action value
                R.append(r)                                 
                R_t.append(np.mean(R))
                rollingReg.append(self.regret(R, i))

        #muEstim = self.getMeans(REstim)    # estimate of true mean reward
        reg = self.regret(R, iters)
        return [reg, R_t, rollingReg]                   # return pseduo-regret and average reward

    def epsGreedy(self, eps, iters):
        Q = Q = [np.random.normal(x,1) for x in self.mu]    # First try all the bandits once
        n = [1]*self.numBandits     # each bandit has been pulled once
        REstim = np.zeros((self.numBandits, iters)) # to estimate the mean of reward distributions
        R = []
        R_t = []
        rollingReg = []

        for i in range(11, iters):          # starting from 11th pull
            if random.random() < eps:       # with probability less than eps, choose action randomly
                a = np.random.randint(1, self.numBandits+1)
                r = np.random.normal(self.mu[a-1], 1)
                REstim[a-1][i] = r 
                n[a-1] += 1
                Q[a-1] = Q[a-1] + ((r- Q[a-1])/n[a-1])
                R.append(r)
                R_t.append(np.mean(R))
                rollingReg.append(self.regret(R, i))
            else:                           # act greedily
                a = np.argmax(Q)
                r = np.random.normal(self.mu[a], 1)
                REstim[a][i] = r
                n[a] += 1
                Q[a] = Q[a] + ((r- Q[a])/n[a])                     # update Q action value
                R.append(r)
                R_t.append(np.mean(R))
                rollingReg.append(self.regret(R, i))

        #muEstim = self.getMeans(REstim)    # estimate of true mean reward
        reg = self.regret(R, iters)
        return [reg, R_t, rollingReg]

    def ucb(self, c, iters):
        Q = [np.random.normal(x,1) for x in self.mu]    # First try all the bandits once
        n = [1]*self.numBandits     # each bandit has been pulled once
        REstim = np.zeros((self.numBandits, iters)) # to estimate the mean of reward distributions
        R = []
        R_t = []
        rollingReg = []
        
        for i in range(11, iters):
            ucb_Q = []
            for bandit in range(0, self.numBandits):
                ucb_Q.append(Q[bandit]+c*np.sqrt(np.log(i)/n[bandit]))
            a = argmax(ucb_Q)
            r = np.random.normal(self.mu[a], 1)
            n[a] += 1
            Q[a] = Q[a] + ((r- Q[a])/n[a])
            R.append(r)
            R_t.append(np.mean(R))
            rollingReg.append(self.regret(R, i))
        
        #muEstim = self.getMeans(REstim)    # estimate of true mean reward
        reg = self.regret(R, iters)         
        return [reg, R_t, rollingReg]

    def softMax(self, alpha, iters):
        H = [np.random.normal(x,1) for x in self.mu]    # pull all arms and generate preference
        R = []
        R_t = []
        rollingReg = []
            
        for i in range(0, iters):
            p = []
            # update pi
            for i in range(0, len(H)):
                p.append(np.exp(H[i])/(sum([np.exp(x) for x in H])))

            # choose best pi
            A = argmax(p)
            r = np.random.normal(self.mu[A], 1)
            R.append(r)
            R_t.append(np.mean(R))
            rollingReg.append(self.regret(R, i))
            # update H
            for j in range(0, len(H)):
                if j == A:
                    H[A] = H[A] + alpha*(r - np.mean(R))*(1-p[A])
                else:
                    H[j] = H[j] - alpha*(r - np.mean(R))*p[j]
        
        reg = self.regret(R, iters)
        return [reg, R_t, rollingReg]
        


    
        