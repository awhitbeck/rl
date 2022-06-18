import random

class RandomAgent :
    def __init(self):
        self.dummy=None

    def play(self,state):
        r = random.randint(0,8)
        while not state[r]==0 :
            r = random.randint(0,8)
        return r
