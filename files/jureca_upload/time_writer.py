import evac as ev
import numpy as np
import datetime
import os

time_file = "time_keeper.txt"
if 1 - os.path.isfile(time_file):
    file = open(time_file,"w")

else:
    file = open(time_file,"a")

file.write(str(datetime.datetime.now()) + " \n")
file.close()
print(datetime.datetime.now())


