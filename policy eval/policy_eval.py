from grid import Grid
import numpy as np
import matplotlib as plt

class Action:

    def __init__(self, type = None, movement = None) -> None:
        self.type = type
        self.movement = movement
    
    def setType(self, a):
        self.type = a

    def getType(self):
        return self.type

    def setMove(self, b):
        self.movement = b

    def getMove(self):
        return self.movement

def sys_dyn(s, a):
    pass

def policy_eval(g, delta):
    
    #init value randomly for each state
    for key in g:
        g[key].append(0)
    
    Delta = 100
    while Delta > 0.01:
        for key in g:
            temp = g[key][1]

            
def genRandPolicy(g):
    types = ['u', 'd', 'r', 'l']
    a = []
    for type in types:
        a.append(Action(type))

    for type in types:
        if key == 'u':
            a[key] = (-1,0)
        elif key == 'd':
            a[key] = (1,0)
        elif key == 'r':
            a[key] = (0,1)
        elif key == 'l':
            a[key] = (0,-1)

    return a

def main():
    g = Grid(); grid = g.createGrid()
    delta = 1
    policy = genRandPolicy(grid)
    #policy_eval(grid, policy, delta)

if __name__ == '__main__':
    main()