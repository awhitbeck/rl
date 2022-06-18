import datetime
import matplotlib.pyplot as plt
import random 
import pickle
from utils import *
from RandomAgent import *
from RLAgent import *
from tictactoe import *

n_batches=500
a = RLAgent()
a.play(n_batches) ## 100 batches of 1000 episodes

plt.plot(range(n_batches),a.average_reward)
plt.grid()
plt.xlabel("Batch number")
plt.ylabel("Average reward")
plt.savefig("TDL_tictactoe.png")
plt.show()

f = open("Models/qfunc_"+str(datetime.datetime.now()).replace(' ','_')+".pkl","wb")
pickle.dump(a.q_function,f)
f.close()

f = open("Models/vfunc_"+str(datetime.datetime.now()).replace(' ','_')+".pkl","wb")
pickle.dump(a.value_function,f)
f.close()
