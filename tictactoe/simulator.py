
import matplotlib.pyplot as plt
from RLAgent import *
from RandomAgent import *
from tictactoe import *

n_batches=500
g = Game(RandomAgent(Game(None)))
a = RLAgent(g)
a.train(n_batches) ## 100 batches of 1000 episodes

plt.plot(range(n_batches),a.average_reward)
plt.grid()
plt.xlabel("Batch number")
plt.ylabel("Average reward")
plt.savefig("TDL_tictactoe.png")
plt.show()

a.save('500')
