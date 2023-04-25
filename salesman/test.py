from RLAgent import *
from Game import *
import time

g = Game()
r = RLAgent(g)
r.gamma=0.9
r.train(300)
#r.save('first')
r.displayQ()
g.Reset()
g.Print()
