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

    def __init__(self, loc = None, type = None, value = 0) -> None:
        self.loc = loc
        self.type = type
        self.value = value

class Grid(Block):
    width = typed_property('width', int)
    height = typed_property('height', int)
    blocked = typed_property('blocked', (float, float))
    trap = typed_property('trap', (float, float))
    goal = typed_property('goal', (float, float))
    start = typed_property('start', (float, float))

    def __init__(self) -> None:
        self.width = 4
        self.height = 3
        self.grid = []
        self.blocked = (1, 1)
        self.trap = (1, 3)
        self.goal= (0, 3)
        self.start = (2, 0)

    def createGrid(self):
        
        for i in range(0, self.height):
            #for j in range(self.width-1, -1, -1):      # start from goal
            for j in range(0, self.width):              # start from other place
                self.grid.append(Block((i,j)))

        for block in self.grid:  
            if block.loc == self.blocked:
                block.type = 'blocked'
            elif block.loc == self.trap:
                block.type = 'trap'
            elif block.loc == self.goal:
                block.type = 'goal'
            elif block.loc == self.start:
                block.type = 'start'
            else:
                block.type = 'free'

        return self.grid

