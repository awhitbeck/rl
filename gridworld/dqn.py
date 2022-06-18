import random
import os
import numpy as np
from collections import deque
from statistics import mean
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(24,input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(24,activation = 'relu'))
        model.add(Dense(self.action_size, activation='relu'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        #print("REPLAYING")
        minibatch = random.sample(self.memory, batch_size)
        for state,action,reward,next_state,done in minibatch:
            #print("state shape: {}, next_state shape: {}".format(state.shape,next_state.shape))
            target = reward
            if not done :
                target = reward + self.gamma*np.amax(self.model.predict(next_state))
            #print("predict")    
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #print("fit")
            self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min :
                self.epsilon*=self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

class Game:

    def __init__(self,imagex,imagey,debug=False):
        self.image=None
        self.imagex=imagex
        self.imagey=imagey
        self.rng = np.random.default_rng(12345)
        self.side=None
        self.xpos=None
        self.ypos=None
        self.debug=debug
        
    def print_image(self):
        for row in self.image:
            print(row)
        
    def state_size(self):
        return self.imagex*self.imagey

    def action_size(self):
        return 4
    
    def generate_scene(self):

        if not self.side%2 :
            self.side+=1
            
        for i in range(-int(self.side/2),int(self.side/2)+1):
            for j in range(-int(self.side/2),int(self.side/2)+1):
                if not self.done(i,j): 
                    self.image[self.xpos+i,self.ypos+j] = 1

        if self.debug:
            print("side: {}, xpos: {}, ypos: {}".format(self.side,self.xpos,self.ypos))
            self.print_image()
        
    def reset(self):

        self.image = np.zeros((self.imagex,self.imagey))
        self.side = 5 #self.rng.integers(low=1,high=5)
        self.xpos = self.rng.integers(low=int((self.imagex-1)/2)-1,high=int((self.imagex-1)/2)+1)
        self.ypos = self.rng.integers(low=int((self.imagex-1)/2)-1,high=int((self.imagex-1)/2)+1)
        self.generate_scene()

        return self.image
    
    def step(self,action):

        """ action can be one of 4 integers (0,1,2,3)
        action 0: move camera up
        action 1: move camera right
        action 2: move camera down
        action 3: move camera left
        """
        if not action in range(4):
            print("ERROR: INVALID ACTION")
            return self.image,self.reward(),self.done()
        
        #if action == 0 :
        #    return self.image,self.reward(),self.done()
        
        ## reset image
        for i in range(-int(self.side/2),int(self.side/2)+1):
            for j in range(-int(self.side/2),int(self.side/2)+1):
                if not self.done(i,j): 
                    self.image[self.xpos+i,self.ypos+j] = 0

        ## shift image
        if action == 0 : # shift image down
            self.ypos+=1
        if action == 1 : # shift image left
            self.xpos-=1 
        if action == 2 : # shift image up
            self.ypos-=1
        if action == 3 : # shift image right
            self.xpos+=1

        ## redraw image

        for i in range(-int(self.side/2),int(self.side/2)+1):
            for j in range(-int(self.side/2),int(self.side/2)+1):
                if not self.done(i,j):
                    self.image[self.xpos+i,self.ypos+j] = 1
        
        return self.image,self.reward(),self.done()

    def done(self,i=0,j=0):
        out_bounds = not (self.xpos+i >= 0 and self.xpos+i < self.imagex and self.ypos+j >= 0 and self.ypos+j < self.imagey)
        goal = self.goal()
        return out_bounds or goal

    def goal(self):
        return int((self.imagex-1)/2) == self.xpos and int((self.imagey-1)/2) == self.ypos 

    def reward(self):
        if self.goal() :
                return 0.
        elif self.done():
            return -20.
        else :
            return -1.
            #return -1*np.sqrt((int((self.imagex-1)/2)-self.xpos)**2+(int((self.imagey-1)/2)-self.ypos)**2)

### Play the game!

# constants
imagex=imagey=21
num_episodes=100
batch_size=64
output_dir='model_output/first_agent'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# init game
game = Game(imagex,imagey,False)
state_size=game.state_size()
action_size=game.action_size()
# init agent
agent = Agent(state_size, action_size)

done=False

running_reward = deque(maxlen=5)

for e in range(num_episodes):

    state = game.reset().reshape((1,state_size))
    reward = 0
    for t in range(16):
        action = agent.act(state)
        next_state,reward_temp,done = game.step(action)
        reward+=reward_temp
        #print("reward: {}, xpos: {}, ypos: {}, done: {}".format(reward,game.xpos,game.ypos,done))
        next_state = next_state.reshape((1,state_size))

        #print("preremember: state shape: {}, next_state shape: {}".format(state.shape,next_state.shape))
        agent.remember(state,action,reward,next_state,done)

        state=next_state

        if t % 20 == 0 :
            print("t: {}".format(t))
        
        if done:
            #print('done')
            break

        if len(agent.memory) > batch_size :
            agent.replay(batch_size)

    print("episode {}/{}, score: {}, e: {:.2}".format(e,num_episodes,reward,agent.epsilon))
    running_reward.append(reward)
    if len(running_reward)>=1:
        print("average episode reward: {}".format(mean(running_reward)))
    print("xpos: {}, ypos: {}".format(game.xpos,game.ypos))

    if e%10 == 0 :
        agent.save(output_dir+'/weights_'+'{:04d}'.format(e)+'.hdf5')
