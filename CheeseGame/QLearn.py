import numpy as np


class Agent:
    def __init__(self, position, world_size, actions):
        self.World = world_size
        self.Actions = actions
        self.Memory = np.zeros((self.World[0], self.World[1], len(self.Actions)))
        self.Position = position

    def learn(self):
        i = -1
        while i < 0 or self.Memory[self.Position[0]][self.Position[1]][i] < 0:
            i = np.random.randint(0, len(self.Actions))
        return i, self.Actions[i]

    def decide(self):
        # get bests
        best = self.Memory[self.Position[0]][self.Position[1]][0]
        bests = [0]
        for i in range(1, len(self.Actions)):
            val = self.Memory[self.Position[0]][self.Position[1]][i]
            if val > best:
                best = val
                bests = [i]
            elif val == best:
                bests.append(i)

        # get random best option
        i = np.random.choice(bests)
        return i, self.Actions[i]

    def remember(self, position, action_index, value, learning_rate=0.1):
        self.Memory[self.Position[0]][self.Position[1]][action_index] = value + learning_rate*np.max(self.Memory[position[0]][position[1]])
        self.Position = position

    def blocked(self, action_index, value):
        self.Memory[self.Position[0]][self.Position[1]][action_index] = value

    def __str__(self):
        print(self.Position, self.Memory)