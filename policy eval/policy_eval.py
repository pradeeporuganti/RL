from grid import Block, Grid, typed_property
import operator
import tkinter as tk
 
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

    # going into the terminal state
    if s.type == 'trap':
        s_next.loc = (-1,-1)
        s_next.value = 0
        s_next.type = 'terminal'
    elif s.type == 'goal':
        s_next.loc = (-1,-1)
        s_next.value = 0
        s_next.type = 'terminal'
    # hitting the walls
    elif s_next.loc[0] < 0 or s_next.loc[0] >= map_.height or\
     s_next.loc[1] < 0 or s_next.loc[1] >= map_.width:
        s_next.loc = s.loc      # bounces back
        s_next.value = s.value
    # hitting the block
    elif s_next.loc == map_.blocked:
        s_next.loc = s.loc      # bounces back
        s_next.value = s.value
    # into the trap
    elif s_next.loc == map_.trap:
        s_next.value = -10      # gets -10 for the trap
        s_next.type = 'trap'
    # reaching the goal
    elif s_next.loc == map_.goal:
        s_next.value = 10       # gets 10 for the trap
        s_next.type = 'goal'

    return s_next

def policy_eval(map_, g, policy, delta):
    
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
    print(iter)
    return g
            
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
        tk.Label(master, text=str(block.value)).grid(row=block.loc[0], column=block.loc[1])
    
    # The mainloop
    tk.mainloop()

def main():
    g = Grid(); grid = g.createGrid()
    delta = 1
    policy = genRandPolicy()
    grid = policy_eval(g, grid, policy, delta)
    showgrid(grid)

    for block in grid:
        print(block.value)

if __name__ == '__main__':
    main()