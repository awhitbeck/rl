# Code for tic-tac-toe

# Game simulator
`tictactoe.py` contains a class that implements the game of tic-tac-toe.
It has two internal representations of the state.  The first is an list
of 9 elements, which represent the unrolled 3x3 game board.
The data for the unrolled board is store in the `board` data members.
In this
representation zero is used to denote an empty space on the board, 1 is
used to denote where Xs are, 2 is used to denote where Os are.  The
computer is always Os.  Currently, this came only has a rudimentary
AI, which chooses among the open space randomly.  The second representation
of the board, that is more condusive to building look-up tables, is
an integer that corresponds to interpreting `board` as digits of a
base-3 number, when converted to base-10.  This representation is
store in the `state` data member.  There are functions to encoode
(convert the board to the state) and decode (convert the state to the
board) the state:

`encodeState()`

`decodeState()`

The first method makes `Game.state` consistent with `Game.board` and
the second is the vice versa.  If you would like to view a state, you
can set it manually and use the `print()` method, but you must first
decode the state.


You can play the computer interactively using the `interactive_play()`
method.  You will be asked to give a space to place an X, you should
give the index of the unrolled board.  In other words,

|index |  row,column    |
|------------|----------|
|   0  | 0 , 0          |
|   1  | 0 , 1		|
|   2  | 0 , 2		|
|   3  | 1 , 0		|
|   4  | 1 , 1		|
|   5  | 1 , 2		|
|   6  | 2 , 0		|
|   7  | 2 , 1		|
|   8  | 2 , 2		|

You can also play the computer through `step()`.  This method takes
an action, evolves the game forward and returns the new state and the
reward associated with the action.  The action space for tictactoe is
technically dependent on the state. However, in this implementation of
the game, we have considered the action space to be fixed.  An action
is an integer between [0,8], and represents where an X should be placed
in the list representation of the board, `Game.board`.  If a user chooses
an action that is invalid, i.e. plays a space that has already been
taken, the game returns the same state and a reward of -10.  If an
action results in a game ending in a win, loss, or tie for the user,
a reward of 1,-1, or 0 is returned, respectively.
If an action does not end the game, a reward of 0 is returned.

If the user would like to check if the game has ended, they can call
`Game.win()` which will return:
-1: if a draw has occured,
 1: if Xs won,
 2: if Os won,
 0: if no one has won.

The method `Game.reset()` can be used to reset the game.  This method will
clear the board and with 50% probability let the computer play first.  

# Benchmark Reinforcement (RL) Agents

Currently only one benchmark RL Agents has been developed for learning
tic-tac-toe. This agent implements a TD-$\lambda$ type algorithm. See
`RLAgent` defined in `RLAgent.py`.  

This class maintains both a value 
function, `RLAgent.value_function`, and a Q value function, 
`RLAgent.q_function`; both are dictionaries whose keys are states of
the game. The vlues of th `RLAgent.value_function` is a list of two elements, 
one maintains the average of the *discounted return*, averaged over all actions 
taken from this state, the other maintains the number of times a state was 
transitioned out of.  The *discounted return* is given by: 

$G_t = \Sigma_{k=0}^{k=\inf}\gamma^kR_{t+k+1}$

Where the subscript refers to a fix time step, $R$ is the reward at a given time 
step and $\gamma$ is the discount factor.  The values of `RLAgent.q_function`
are another dictionary that is keyed on actions taken from the state.  The values 
of these subdictionaries are the average *discounted return*.  This structure 
makes it easy to search for the maximum of the q-function, $q(s,a)$.

The agent perform N batches of M games.  It tracks it average end-of-game reward
per batch and reports that to the standard output.  The action taken by the agent 
is either a random action or based on a policy.  The policy is given by: 

$\pi(s) = argmax_a q(s,a)$.

This random choice enforces some amount of exploration.  The probability of 
a random action is given by `RLAgent.lambda`.  This probability is initiallly
1 and is decayed every batch by `RLAgent.decay`.  

## Performance of agent

This agent has been trained over several hundred batch of 10k games when 
playing a simple AI that makes random selections.  When the agent always
plays first, it was found that the average end-of-game reward was ~0.98.  When
players randomly go first with equal probability, the agent acheived an 
average end-of-game reward of .88.  The Figure below shows the average 
end-of-game reward versus batch number. 

![This is an image](https://github.com/awhitbeck/rl/blob/main/tictactoe/TDL_tictactoe.png)
