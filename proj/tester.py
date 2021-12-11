import gym
import envs
import numpy as np
from RL_main import Agent
from casadi import *
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.01, input_dims=[5], gamma=0.99, layer1_size=256,
                  layer2_size=256)
    env = gym.make('RLProj-v0')
    score_history = []
    num_episodes = 1000
    actions = np.array([0])

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        observation = np.array(observation).reshape(1, 5)
        #print(observation)
        #step = 0
        while not done:
            action = np.array(agent.choose_action(observation)).reshape(1, 1)
            #print(action)
            #actions = np.concatenate((actions, action), axis=0)
            observation_, reward, done, _ = env.step(action)
            #print(observation_)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
            #step += 1
            #print(observation, reward, step)
        print('episode ', i, 'score %.2f' % score)

    func = external('F', './vehd.so')
    done = False
    #observation = env.reset()
    #print(observation)
    #observation = env.reset()
    #observation = np.array(observation).reshape(1, 5)
    #print(observation)
    observation = env.reset()
    observation = np.array(observation).reshape(1, 5)
    #observation = [[0, 0, 0, 20, 20]]
    state = observation
    while not done:
        action = np.array(agent.choose_action(observation)).reshape(1, 1)
        #print(action)
        observation_, reward, done, info = env.step(action)
        #print(observation_, reward, done)
        observation = observation_
        state = np.concatenate((state, np.reshape(observation_, (1, 5))), axis=0)

    plt.figure(1)
    plt.plot(state[:, 4], state[:, 3])
    plt.show()









