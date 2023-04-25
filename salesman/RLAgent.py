import random
import pickle
import datetime
import numpy as np

class RLAgent:
    def __init__(self,game):
        self.dummy=None
        self.value_function={}       # value function v(s)
        self.q_function={}           # action value function q(s,a)
        self.game=game
        self.current_state=0
        self.previous_state=0
        self.gamma=0.5              # discount factor for discounted rewards
        self.lambda_=1.0             # explore/exploit parameter
        self.decay=0.99              # for reducing lambda
        self.batch_size=100        # number of episodes per batch
        self.average_reward=[]

    """
    write value and Q function to pickle file
    """
    def save(self,tag):
        f = open("Models/qfunc_"+tag+"_"+str(datetime.datetime.now()).replace(' ','_')+".pkl","wb")
        pickle.dump(self.q_function,f)
        f.close()

        f = open("Models/vfunc_"+tag+"_"+str(datetime.datetime.now()).replace(' ','_')+".pkl","wb")
        pickle.dump(self.value_function,f)
        f.close()

    """
    load value and Q function from pickle file
    """
    def load(self,vfunc_file_name,qfunc_file_name):
        f = open(vfunc_file_name,'rb')
        self.value_function = pickle.load(f)
        f.close()

        f = open(qfunc_file_name,'rb')
        self.q_function = pickle.load(f)
        f.close()
        
    """
    implemention of an iterative update of the value function
    """
    def iterate_value_function(self,state,reward):
        if not state in self.value_function:
            self.value_function[state] = [reward,1]
        else :
            value = self.value_function[state][0]
            count = self.value_function[state][1]
            self.value_function[state] = [ value + (1./count)*(reward - value) , count+1 ]

    """
    implementation of an iterative update of the action value function 
    """
    def iterate_q_function(self,state,action,reward):
        if not state in self.q_function:
            self.q_function[state] = {action:[reward,1]}
        else :
            if not action in self.q_function[state] :
                self.q_function[state][action]=[reward,1]
            else:
                value = self.q_function[state][action][0]
                count = self.q_function[state][action][1]
                self.q_function[state][action] = [ value + (1./count)*(reward - value) , count+1 ]
    
    """
    returns a random selection of possible actions
    """
    def random_action(self,state):
        #pick an action at random
        #action = random.randint(0,len(self.game.ACTIONS)-1)
        action = random.randint(0,len(self.game.UNVISITED_SITES)-1)
        return self.game.FindIndex(self.game.SITES[action])
        #return self.game.ACTIONS[action]

    """
    returns the maximum value of Q for a given state
    """
    def argMaxQ(self,state):
        max_action=-9
        max_q_value=-9999
        for key in self.q_function[state]:
            if self.q_function[state][key][0] > max_q_value:
                max_q_value=self.q_function[state][key][0]
                max_action=key
        return max_action

    """
    returns the action that has the maximum Q value
    for a given state
    """
    def MaxQ(self,state):
        return self.q_function[state][self.argMaxQ(state)][0]

    """
    returns an action that is greedy
    """
    
    """ 
    returns an action based on a policy
    policy is given by \pi(s) = argmax_{a} q(s,a)
    """
    def policy_action(self,state):
        #self.game.print()
        if not state in self.q_function:
            return self.random_action(state)
        else:
            #print(self.q_function[state])
            return self.argMaxQ(state)

    """
    returns an action given a state based on an 
    epsilon-greedy policy
    """
    def play(self):
        #self.game.state = state
        #self.game.decodeState()
        if random.randint(0,10000) < 10000*self.lambda_ :
            return self.random_action(self.game.EncodeState(self.game.state))
        else :
            return self.policy_action(self.game.EncodeState(self.game.state))

    """
    record user actions
    """
    def record(self):
        self.game.Reset()
        self.game.Print()
        while not self.game.Win() :
            while True: 
                action = input("Choose a direction to move: ")
                action = action.lower()
                if action == 'q' : return
                if action in self.game.ACTIONS : break
                else : print('invalid...')
            previous_state = self.game.EncodeState(self.game.state)
            current_state,reward = self.game.Step(action)
            print('previous state: ',previous_state,' current state: ', current_state,' reward: ',reward)
            if current_state in self.value_function :
                print('updating q-function for old state')
                self.iterate_value_function(previous_state,reward+self.gamma*self.value_function[current_state][0])
                self.iterate_q_function(previous_state,action,reward+self.gamma*self.MaxQ(current_state))
            else :
                print('updating q-function for new state')
                self.iterate_value_function(previous_state,reward)
                self.iterate_q_function(previous_state,action,reward)
            self.game.Print()
        
    """ 
    function for training agent
    """
    def train(self,n_batches):
        for batch in range(n_batches):
            print("batch ",batch," of ",n_batches)
            self.average_reward.append(0)
            for episode in range(self.batch_size):
                reward=0
                self.game.Reset()
                self.current_state=self.game.EncodeState(self.game.state)
                self.previous_state=-1
                #print("- - - - - - - - ")
                #self.game.Print()
                n_actions = 0 
                while not self.game.Win() and n_actions < 100 :
                    n_actions+=1
                    #print('n_actions',n_actions)
                    action=self.play()
                    #update state of game
                    self.previous_state = self.current_state
                    self.current_state,reward=self.game.Step(action)
                    #print("reward: ",reward," previous state: ",self.previous_state)
                    if self.current_state in self.value_function : 
                        self.iterate_value_function(self.previous_state,reward+self.gamma*self.value_function[self.current_state][0])
                        #self.iterate_q_function(self.previous_state,action,reward+self.gamma*self.value_function[self.current_state][0])
                        self.iterate_q_function(self.previous_state,action,reward+self.gamma*self.MaxQ(self.current_state))
                    else :
                        self.iterate_value_function(self.previous_state,reward)
                        self.iterate_q_function(self.previous_state,action,reward)

                self.average_reward[-1]+=(1./(episode+1))*(reward-self.average_reward[-1])
                
            # at end of batch update lambda
            self.lambda_*=self.decay
            print('avg reward:',self.average_reward[-1],' lambda: ',self.lambda_)

    def dumpQ(self):
        for key in self.q_function :
            print(" - - - - - - - - - - - - - - - - - - - - - - - -")
            print("state: ",key)
            #self.game.state = key
            #self.game.DecodeState(key)
            #self.game.Print()
            print(self.q_function[key])

    def displayQ(self):
        mat=[]
        for i in range(self.game.WORLD_SIZE):
            mat.append(['']*self.game.WORLD_SIZE)
        for i in range(self.game.WORLD_SIZE*self.game.WORLD_SIZE):
            pos=self.game.DecodeState(i)
            if i in self.q_function :
                #print('best action ',self.argMaxQ(i))
                #print('location: ',np.argwhere(self.game.ACTIONS==self.argMaxQ(i)))
                #print('pos: ',pos)
                #print('state: ',i,' qmax: ',self.argMaxQ(i),' ',self.game.ACTION_LABELS[np.argwhere(self.game.ACTIONS==self.argMaxQ(i))[0][0]])
                mat[-pos[1]-1][pos[0]]=self.game.ACTION_LABELS[np.argwhere(self.game.ACTIONS==self.argMaxQ(i))[0][0]]
            else :
                mat[-pos[1]-1][pos[0]]='\u25B6'
        for i in range(self.game.WORLD_SIZE):
            print(mat[i])
    
