from Game import *
from RLAgent import *

g = Game()
r = RLAgent(g)
r.lambda_=0.0
r.load("Models/vfunc_first_2022-12-16_17:16:16.341854.pkl","Models/qfunc_first_2022-12-16_17:16:16.106968.pkl")

g.Reset()
g.Print()

