import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

class Game:

    def __init__(self):
        plt.ion()
        self.state = np.array([0,0]) ## x,y coordinate
        self.WORLD_SIZE = 200
        df = pd.read_csv('holes.txt', sep=',', header=None)
        x = df[0]
        x = np.add(x,-min(x))
        y = df[1]
        y = np.add(y,-min(y))
        temp = list(map(list,list(zip(x,y))))
        self.SITES = np.array(temp)
        #self.SITES = np.array([[60,140],
        #                      [60,60],
        #                      [140,60],
        #                      [140,140]])
        self.UNVISITED_SITES = self.SITES
        self.ACTIONS = np.array(np.arange(0,len(self.SITES),1))
        #print('actions: ',self.ACTIONS)
        
    def EncodeState(self,state):
        return self.WORLD_SIZE*state[0]+state[1]

    def DecodeState(self,code):
        return [int(code/self.WORLD_SIZE),code%self.WORLD_SIZE]

    def Reset(self):
        #self.state = self.DecodeState(random.randint(0,self.WORLD_SIZE*self.WORLD_SIZE))
        self.state = np.array([0,0])
        self.UNVISITED_SITES = self.SITES

    def Print(self):
        temp_locs=self.UNVISITED_SITES
        temp_locs=np.append(temp_locs,[self.state],axis=0)
        temp_locs=np.append(temp_locs,[self.state],axis=0)
        t = np.transpose(temp_locs)
        plt.hist2d(x=t[0],y=t[1],bins=[np.arange(0,self.WORLD_SIZE+1,1),np.arange(0,self.WORLD_SIZE+1,1)])
        plt.show(block=False)

    def FindIndex(self,state):
        dists = self.SITES - state
        dist = []
        for d in dists :
            dist.append(int(np.sqrt(d[0]*d[0]+d[1]*d[1])))
        #print('index ',dist.index(0))
        return dist.index(0)
        
    def Step(self,action):
        previous_state = self.state
        self.state = self.SITES[action]
        disp = self.state-previous_state
        reward = -np.sqrt(disp[0]*disp[0]+disp[1]*disp[1])
        dist = self.UNVISITED_SITES - self.SITES[action]
        dist = list(map(lambda x : int(np.sqrt(x[0]*x[0]+x[1]*x[1])),dist))
        #print('dist: ',dist)
        if 0 in dist : 
            #print('index: ',dist.index(0))
            self.UNVISITED_SITES = np.delete(self.UNVISITED_SITES,dist.index(0),axis=0)
            return self.EncodeState(self.state), 100. + reward
        else:
            return self.EncodeState(self.state), + reward

    def Win(self):
        return len(self.UNVISITED_SITES) == 0

def InteracativePlay():
    g = Game()
    g.Reset()
    g.Print()
    while not g.Win():
        while True:
            action = input("Choose a direction to move: ")
            #action = action.lower()
            action = int(action)
            if action == 'q' : return
            if action in g.ACTIONS : break
            else : print('invalid...')
        new_state,reward = g.Step(action)
        g.Print()
    print("congrats")

