def typed_property(name, expected_type):
    storage_name = '_' + name
    
    @property
    def prop(self):
        return getattr(self, storage_name)
    
    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)
        return prop

class Block:   
    loc = typed_property('loc', (float, float))
    type = typed_property('type', str)
    value = typed_property('value', float)

    def __init__(self, loc = None, type = None, Qvalue = {}) -> None:
        self.loc = loc
        self.type = type
        self.Qvalue = Qvalue

class Grid(Block):
    width = typed_property('width', int)
    height = typed_property('height', int)
    blocked = typed_property('blocked', (float, float))
    trap = typed_property('trap', (float, float))
    goal = typed_property('goal', (float, float))
    start = typed_property('start', (float, float))

    def __init__(self) -> None:
        self.width = 12
        self.height = 4
        self.grid = []
        self.blocked = (100,100)
        self.trap = [(self.height-1,i) for i in range(1,self.width-1)] 
        self.goal= (self.height-1, self.width-1)
        self.start = (self.height-1, 0)
        self.terminal = (-1, -1)

    def createGrid(self):

        for i in range(0, self.height):
            #for j in range(self.width-1, -1, -1):      # start from goal
            for j in range(0, self.width):              # start from other place
                self.grid.append(Block((i,j)))
            self.grid.append(Block(self.terminal))

        for block in self.grid:  
            if block.loc in self.blocked:
                block.type = 'blocked'
            elif block.loc in self.trap:
                block.type = 'trap'
            elif block.loc == self.goal:
                block.type = 'goal'
            elif block.loc == self.start:
                block.type = 'start'
            elif block.loc == self.terminal:
                block.type = 'terminal'
            else:
                block.type = 'free'

        return self.grid