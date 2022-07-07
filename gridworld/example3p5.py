
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

""" 
This code implements the gridworld game described in Example 3.8 
of Sutton and Barto.  I have removed all code that implements a 
solution to the game.

If you want to play it just to see how the environment dynamics work 
you can load this, initialize a instance of the `Game` class and call
the `InteractivePlay()` method. 

Unlike the original code this started from, the game maintains its 
own state: `Game.state`.  There are also helper methods to generalize
RL agent code, namely `Game.ActionSpace()`, which returns the list of 
actions the agent can take.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

class Game:
    def __init__(self):
        self.state=[0,0]
        self.WORLD_SIZE = 5
        self.A_POS = [0, 1]
        self.A_PRIME_POS = [4, 1]
        self.B_POS = [0, 3]
        self.B_PRIME_POS = [2, 3]

        # actions are n: north, s: south, e: east, w: west
        self.ACTIONS = ['n','s','e','w']
        self.ACTION_STEPS = [np.array([-1, 0]),
                            np.array([1, 0]),
                            np.array([0, 1]),
                            np.array([0, -1])]
        self.ACTIONS_FIGS=[ '←', '↑', '→', '↓']

    def EncodeState(self,state):
        return self.WORLD_SIZE*state[0]+state[1]

    def DecodeState(self,code):
        return [int(code/self.WORLD_SIZE),code%self.WORLD_SIZE]
        
    def ActionSpace(self):
        return self.ACTIONS
    
    def Reset(self):
        self.state = [0,0]

    def Print(self):
        world = np.zeros((self.WORLD_SIZE,self.WORLD_SIZE))
        x,y = self.state
        world[x][y] = 1
        for row in world:
            print(" ".join(str(row)))
            
    def Step(self,action):
        #print('starting step',self.state,' action: ',action)
        #if in position A move to A' return a reward of 10
        if self.state == self.A_POS:
            self.state = self.A_PRIME_POS
            return self.A_PRIME_POS, 10
        #if in position B move to B' return a reward of 5
        if self.state == self.B_POS:
            self.state = self.B_PRIME_POS
            return self.B_PRIME_POS, 5
        
        #if in neither A or B, move according to action
        step = self.ACTION_STEPS[self.ACTIONS.index(action)]
        next_state = (np.array(self.state) + step).tolist()
        x, y = next_state
        ## penalize player for trying to move across boundary
        if x < 0 or x >= self.WORLD_SIZE or y < 0 or y >= self.WORLD_SIZE:
            reward = -1.0
            next_state = self.state
        else:
            reward = 0
        self.state = next_state
        return next_state, reward

    def InteractivePlay(self):
        g = Game()
        action=""
        g.Reset()
        g.Print()
        while True:
            while True: 
                try :
                    action = input("Choose direction to move [n,s,e,w] (q to quite):")
                    action = action.lower()
                    if action == 'q' : return 
                    if action in g.ACTIONS : 
                        break
                    else :
                        print('invalid...')
                except ValueError:
                    print('invalid...')
            new_state,reward = g.Step(action)
            print('reward: ',reward)
            g.Print()
