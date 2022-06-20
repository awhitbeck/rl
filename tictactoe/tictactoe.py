import random

class Game:

    def __init__(self,agent):
        self.boardSize=3
        self.board = [0]*self.boardSize**2
        self.state = 0
        self.first_prob = 0.5
        self.game_ai = agent
        self.win_combo = {"H1":[0,1,2],
                          "H2":[3,4,5],
                          "H3":[6,7,8],
                          "V1":[0,3,6],
                          "V2":[1,4,7],
                          "V3":[2,5,8],
                          "D1":[0,4,8],
                          "D2":[2,4,6]
                          }

    """ 
    convert the game board to the state ID 
    The state ID is found by mapping the board to
    a base-10 integer by interpreting each spot on the board
    as a digit in a base-3 integer.  Empty: 0; X: 1; O: 2.  
    """
    def encodeState(self):
        total=0
        for i in range(self.boardSize**2):
            total+=(3**i)*self.board[i]
        self.state = total
        return total

    """ convert the state ID to a game board """
    def decodeState(self):
        tempState=self.state
        for i in range(9):
            self.board[i] = tempState%3
            tempState = tempState//3
        return self.board

    """ 
    reset board to all blanks
    with probability 1-self.first_prob have game AI play first
    """
    def reset(self):
        #print('Game.reset()')
        self.board = [0]*9
        self.encodeState()
        if random.randint(0,1000) >= 1000*self.first_prob :
            action = self.game_ai.play(self.state)
            self.board[action]=2
        self.encodeState()
            
    """function that returns the status of the 
       game.
       -1: draw
       1: Xs won
       2: Os won
       0: no one has won
    """
    def win(self):
        #check if one of the winning combinations has been achieved
        for key,combo in self.win_combo.items() :
            if self.board[combo[0]] == self.board[combo[1]] and self.board[combo[0]] == self.board[combo[2]] and not self.board[combo[0]] == 0 :
               return self.board[combo[0]]

        # check if board has empty space (0)
        full=True
        for i in range(9):
            if self.board[i] == 0  :
                full = False
                break

        #if full game is null
        if full : 
            return -1
        else :
            return 0


    """ print the board to the standard output"""
    def print(self):
        print("STATE:",self.state)
        for i in range(3):
            print("{0} {1} {2}".format(self.board[i*3],self.board[i*3+1],self.board[i*3+2]))
        
    """ start a game in which the user plays a random agent"""
    def interactive_play(self):
        self.reset()

        while(self.win()==0):
            self.print()
            action = int(input("What position do you want to ploy?"))
            while not self.board[action]==0 :
                action = int(input("Can't play there... Try again:"))

            self.board[action]=1
            self.encodeState()
            if not self.win()==0: break

            #allow game ai to play...
            r=self.game_ai.play(self.state)
            self.board[r]=2
            self.encodeState()

        self.print()
        end_game_result = self.win()
        if end_game_result == 1:
            print("You won!")
        elif end_game_result == -1:
            print("...draw.  Too bad.")
        else :
            print("You lose!")

    """ 
    function in auto play to take an action and 
    evolve the state of the environment forward

    If an invalid action is taken return -10 reward and 
    leave game in its current state

    if game won, send back +1 reward
    if game lost, send back -1 reward
    if game is a draw, send back 0 reward

    return: (the reward , the new state of the game)
    """
    def step(self,action):
        if not self.board[action] == 0 :
            #self.print()
            return -10,self.state
        else:
            self.board[action]=1
            self.encodeState()
            #print("Game.step()")
            #self.print()
            if self.win()==0: 
                #allow game ai to play...
                r=self.game_ai.play(self.state)
                self.board[r]=2
                self.encodeState()

        #self.print()
            
        end_game_result = self.win()
        if end_game_result == 1 :
            return 1,self.state
        elif end_game_result == -1:
            return 0,self.state
        elif end_game_result == 2 :
            return -1,self.state
        else:
            return 0,self.state

