import random

class RandomAgent :
    def __init__(self,game):
        self.dummy=None
        self.game=game
        
    def play(self,state):
        #print("RandomAgent.play -- state:",state)
        self.game.state = state
        self.game.decodeState()
        #self.game.print()
        r = random.randint(0,8)
        while not self.game.board[r]==0 :
            r = random.randint(0,8)
        return r
