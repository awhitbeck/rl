import random
from tictactoe import *
import pickle
import datetime

class RLAgent:
    def __init__(self,game,playing_x=True):
        self.dummy=None
        self.value_function={}       # value function v(s)
        self.q_function={}           # action value function q(s,a)
        self.game=game
        self.current_state=0
        self.previous_state=0
        self.gamma=0.9               # discount factor for discounted rewards
        self.lambda_=1.0             # explore/exploit parameter
        self.decay=0.99              # for reducing lambda
        self.batch_size=10000        # number of episodes per batch
        self.average_reward=[]
        self.playing_x=playing_x

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

    def invert_board(self,board):
        print('inverting ',' '.join(map(str,board)))
        new_board=[0]*9
        for i,val in enumerate(board):
            if val == 0 : continue
            elif val == 1 : new_board[i]=2
            elif val == 2 : new_board[i]=1
            else:
                raise ValueError('board can only have values 0, 1, or 2')
        print('new board ',' '.join(map(str,new_board)))
        return new_board
    
    """
    returns a random selection of empty spaces
    """
    def random_action(self,state):
        self.game.state = state
        self.game.decodeState()
        #pick an action at random
        action = random.randint(0,8)
        while not self.game.board[action]==0 :
            action = random.randint(0,8)
        return action

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
    returns an action based on a policy
    policy is given by \pi(s) = argmax_{a} q(s,a)
    """
    def policy_action(self,state):
        self.game.state = state
        self.game.decodeState()
        #self.game.print()
        if not self.playing_x :
            self.game.board = self.invert_board(self.game.board)
            #self.game.print()
            self.game.encodeState()
        if not self.game.state in self.q_function:
            return self.random_action(self.game.state)
        else:
            print(self.q_function[self.game.state])
            return self.argMaxQ(self.game.state)

    """
    returns an action given a state based on an 
    epsilon-greedy policy
    """
    def play(self,state):
        self.game.state = state
        self.game.decodeState()
        if random.randint(0,10000) < 10000*self.lambda_ :
            return self.random_action(state)
        else :
            return self.policy_action(state)

    """ 
    function for training agent
    """
    def train(self,n_batches):
        for batch in range(n_batches):
            print("batch ",batch," of ",n_batches)
            self.average_reward.append(0)
            for episode in range(self.batch_size):
                reward=0
                self.game.reset()
                self.current_state=self.game.state
                self.previous_state=0
                #print("- - - - - - - - ")
                #self.game.print()
                while not self.game.win() :
                    action=self.play(self.current_state)
                    #update state of game
                    self.previous_state = self.current_state
                    reward,self.current_state=self.game.step(action)
                    #print("reward: ",reward," previous statee: ",self.previous_state)
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
            print('avg reward:',self.average_reward[-1])

    def dumpQ(self):
        for key in self.q_function :
            print(" - - - - - - - - - - - - - - - - - - - - - - - -")
            print("state: ",key)
            self.game.state = key
            self.game.decodeState()
            self.game.print()
            print(self.q_function[key])

    
