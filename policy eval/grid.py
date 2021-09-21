class Block:

    def __init__(self, loc, type = None, value = 0) -> None:
        self.loc = loc
        self.type = type
        self.value = value

    def setType(self, a):
        self.type = a

    def setValue(self, b):
        self.value = b

    def getLoc(self):
        return self.loc

    def getType(self):
        return self.type

    def getValue(self):
        return self.value

class Grid(Block):
    
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
            for j in range(0, self.width):
                self.grid.append(Block((i,j)))

        for block in self.grid:  
            if block.getLoc() == self.blocked:
                block.setType('blocked')
            elif block.getLoc() == self.trap:
                block.setType('trap')
            elif block.getLoc() == self.goal:
                block.setType('goal')
            elif block.getLoc() == self.start:
                block.setType('start')
            else:
                block.setType('free')

        return self.grid

def main():
    g = Grid()
    grid = g.createGrid()

if __name__ == '__main__':
    main()

