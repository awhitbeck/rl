from RLAgent import *
from tictactoe import *

#qfunc_500_2022-06-18_15:33:50.160950.pkl
#vfunc_500_2022-06-18_15:33:50.171808.pkl

a = RLAgent(Game(None))
a.load(vfunc_file_name="Models/vfunc_500_2022-06-20_19:12:23.006871.pkl",
       qfunc_file_name="Models/qfunc_500_2022-06-20_19:12:22.994413.pkl"
       )
a.lambda_=0 ### make sure that AI is always play using its policy
a.playing_x=False
g = Game(a)
g.interactive_play()



