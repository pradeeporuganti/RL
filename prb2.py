import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.core.fromnumeric import mean

class Bandit:

    def __init__(self, numBandits):
        self.numBandits = numBandits
        self.genRewards() 

    def genRewards(self):   # Generates random mean for rewards for each bandit
        self.mu = []
        i = 0
        while i < self.numBandits:
            self.mu.append(np.random.normal())
            i += 1
        return self.mu

    def nGreedy(self, N, iters):    # Explore the first N times and then pick greedy action
        self.N = N
        self.iters = iters
        Q = [1]*self.numBandits
        n = [0]*self.numBandits
        R = []
        R_t = []

        for i in range(iters):
            if i < N:
                a = np.random.randint(1, self.numBandits)
                r = np.random.normal(self.mu[a-1], 1)
                n[a] += 1
                R.append(r)
                R_t.append(np.mean(R))
                
                Q[a] += (r + Q[a])/n[a]
            else:
                a = np.argmax(Q)
                r = np.random.normal(self.mu[a], 1)
                R.append(r)
                R_t.append(np.mean(R))
        
        plt.plot(R_t,'k-')
        plt.draw()
        plt.ylabel('avg_reward'); plt.xlabel('iterations')
        #print(Q); print(a)
        

    def epsGreedy(self, eps, iters):
        self.eps = eps
        self.iters = iters
        Q = [0]*self.numBandits
        n = [0]*self.numBandits
        R = []
        R_t = []

        for i in range(iters):
            if random.random() < eps:
                a = np.random.randint(1, self.numBandits)
                r = np.random.normal(self.mu[a-1], 1)
                n[a] += 1
                Q[a] += (r + Q[a])/n[a]
                R.append(r)
                R_t.append(np.mean(R))
            else:
                a = np.argmax(Q)
                r = np.random.normal(self.mu[a], 1)
                R.append(r)
                R_t.append(np.mean(R))
        
        x = [i for i in range(iters)]
        ax = plt.gca()
        ax.plot(x, R_t,'r-')
        
        plt.show()
        #plt.ylabel('avg_reward'); plt.xlabel('iterations')

        #print(Q)
        #print(self.mu)

def main():
    bandits = Bandit(10)
    bandits.nGreedy(200, 1000)
    bandits.epsGreedy(0.2, 1000)

if __name__ == '__main__':
    main()
        


    
        