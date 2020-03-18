import numpy as np


class Agent:
    def __init__(self, position, world_size, actions):
        self.World = world_size
        self.Actions = actions
        # Memory contains actions values for each tile of the world (X x Y x Actions)
        self.Memory = np.zeros((self.World[0], self.World[1], len(self.Actions)))
        self.Position = position

    def decide(self):
        """
        Get Agent next move, based on his Memories.

        :return: [Index of action, Action]
        """
        # get bests options
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
        """
        Remember consequences of action.

        :param position: New position calculated from decision
        :param action_index: Index of selected action
        :param value: Evaluated value
        :param learning_rate: Learning rate
        """
        self.Memory[self.Position[0]][self.Position[1]][action_index] = value + learning_rate*np.max(self.Memory[position[0]][position[1]])
        self.Position = position

    def blocked(self, action_index, value):
        """
        Since we dont want to learn what wall is, we just remember it as constant.

        :param action_index: Index of selected action
        :param value: Evaluated Value
        """
        self.Memory[self.Position[0]][self.Position[1]][action_index] = value

    def __str__(self):
        """
        Prints Agent Position and Memory matrix
        """
        print(self.Position, self.Memory)
