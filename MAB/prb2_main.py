import numpy as np
import matplotlib.pyplot as plt

from prb2_funcs import Bandit

def main():
    bandits = Bandit(10)
    T = 1000
    samples = 1

    # samples and parameters for NGreedy algorithm
    regGreedy = []
    N = np.linspace(10, 100, 11)
    rt_Ngreedy = bandits.nGreedy(50, T)[1]
    regRoll_Ngreedy = bandits.nGreedy(50, T)[2]
    for Ni in N:
        temp = []
        for sample in range(0, samples):
            temp.append(bandits.nGreedy(Ni, T)[0])
        regGreedy.append(np.mean(temp))
    
    plt.subplot(221)
    plt.title('N-greedy')
    plt.plot(N, regGreedy, 'o-')
    plt.xlabel('N'); plt.ylabel('pseudo-Regret')
    plt.grid('on', alpha = 0.4)
    plt.draw()

    # samples and parameters for epsGreedy algorithm
    regEps = []
    eps = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
    rt_eps = bandits.epsGreedy(eps[2], T)[1]
    regRoll_eps = bandits.epsGreedy(eps[2], T)[2]
    for epsi in eps:
        temp = []
        for sample in range(0, samples):
            temp.append(bandits.epsGreedy(epsi, T)[0])
        regEps.append(np.mean(temp))
    
    plt.subplot(222)
    plt.title('eps-greedy')
    plt.plot(eps, regEps, 'o-')
    plt.xlabel('epsilon'); plt.ylabel('pseudo-Regret')
    plt.grid('on', alpha = 0.4)
    plt.draw()

    # samples and parameters for ucb algorithm
    regucb = []
    c = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    rt_ucb = bandits.ucb(c[3], T)[1]
    regRoll_ucb = bandits.ucb(c[3], T)[2]
    for ci in c:
        temp = []
        for sample in range(0, samples):
            temp.append(bandits.ucb(ci, T)[0])
        regucb.append(np.mean(temp))
    
    plt.subplot(223)
    plt.title('UCB')
    plt.plot(c, regucb, 'o-')
    plt.xlabel('c'); plt.ylabel('pseudo-Regret')
    plt.grid('on', alpha = 0.4)
    plt.draw()

    # samples and parameters for ucb algorithm
    regSoftMax = []
    alpha = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    rt_softMax = bandits.softMax(alpha[1], T)[1]
    regRoll_softMax = bandits.softMax(alpha[1], T)[2]
    for alphai in alpha:
        temp = []
        for sample in range(0, samples):
            temp.append(bandits.softMax(alphai, T)[0])
        regSoftMax.append(np.mean(temp))
    
    plt.subplot(224)
    plt.title('Gradient-based')
    plt.plot(alpha, regSoftMax, 'o-')
    plt.xlabel('alpha'); plt.ylabel('pseudo-Regret')
    plt.grid('on', alpha = 0.4)
    plt.draw()

    fig2, ax2 = plt.subplots()
    ax2.plot(rt_Ngreedy)
    ax2.plot(rt_eps)
    ax2.plot(rt_ucb)
    ax2.plot(rt_softMax)
    ax2.legend(['Ngreedy', 'eps-greedy','UCB','grad-bandit'])
    plt.grid('on', alpha = 0.4)
    plt.draw()

    fig3, ax3 = plt.subplots()
    #ax3.plot(regRoll_Ngreedy)
    ax3.plot(regRoll_eps)
    ax3.plot(regRoll_ucb)
    #ax3.plot(regRoll_softMax)
    ax3.legend(['eps-greedy','UCB'])
    plt.grid('on', alpha = 0.4)
    plt.draw()

    plt.show()

if __name__ == '__main__':
    main()
