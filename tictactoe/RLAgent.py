import random
from tictactoe import *

class RLAgent:
    def __init__(self):
        self.dummy=None
        self.value_function={}
        self.q_function={}
        self.game=Game()
        self.current_state=0
        self.previous_state=0
        self.gamma=0.9
        self.lambda_=1.0
        self.decay=0.99
        self.batch_size=10000
        self.average_reward=[]
        
    def iterate_value_function(self,state,reward):
        if not state in self.value_function:
            self.value_function[state] = [reward,1]
        else :
            value = self.value_function[state][0]
            count = self.value_function[state][1]
            self.value_function[state] = [ value + (1./count)*(reward - value) , count+1 ]


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

    def random_action(self):
        #pick an action at random
        action = random.randint(0,8)
        while not self.game.board[action]==0 :
            action = random.randint(0,8)
        return action

    def policy_action(self,state):
        if not state in self.q_function:
            return self.random_action()
        else:
            max_action=-9
            max_q_value=-9999
            for key in self.q_function[state]:
                if self.q_function[state][key][0] > max_q_value:
                    max_q_value=self.q_function[state][key][0]
                    max_action=key
            return max_action
        
    def play(self,n_batches):
        for batch in range(n_batches):
            print("batch ",batch," of ",n_batches)
            self.average_reward.append(0)
            for episode in range(self.batch_size):
                reward=0
                self.game.reset()
                self.current_state=self.game.state
                self.previous_state=0
                #print("- - - - - - - - ")
                while not self.game.win() :

                    if random.randint(0,10000) < 10000*self.lambda_ :
                        action = self.random_action()
                    else :
                        action = self.policy_action(self.current_state)

                    #update state of game
                    self.previous_state = self.current_state
                    reward,self.current_state=self.game.step(action)
                    #print("reward: ",reward," previous statee: ",self.previous_state)
                    if self.current_state in self.value_function : 
                        self.iterate_value_function(self.previous_state,reward+self.gamma*self.value_function[self.current_state][0])
                        self.iterate_q_function(self.previous_state,action,reward+self.gamma*self.value_function[self.current_state][0])
                    else :
                        self.iterate_value_function(self.previous_state,reward)
                        self.iterate_q_function(self.previous_state,action,reward)

                self.average_reward[-1]+=(1./(episode+1))*(reward-self.average_reward[-1])
                
            # at end of batch update lambda
            self.lambda_*=self.decay
            print('avg reward:',self.average_reward[-1])
            
        #print(len(self.value_function))
        #print(self.value_function)
        # for key in self.q_function :
        #     print(" - - - - - - - - - - - - - - - - - - - - - - - -")
        #     print("state: ",key)
        #     self.game.state = key
        #     self.game.decodeState()
        #     self.game.print()
        #     print(self.q_function[key])


