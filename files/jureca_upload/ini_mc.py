import numpy as np
import os
import sys
import evac as ev
#sys.path.insert(1,'/p/project/jias70/jps_jureca/files/jureca_upload')


b_max = 2.0
b_min = 1.2
b_step = 0.01
dig = 3
#i_start = 0
#i_final = 100

size = (b_max-b_min)/b_step
runstep = 0.1
#print(runstep/b_step)
run_jump = int(runstep/b_step)
#i_step = 1
#irange = np.arange(i_start,i_final,i_step)
brange = np.arange(b_min,b_max,b_step)
#print(brange.shape[0])
np.save("ini_b.npy", brange)
brange = np.array([round(i, dig) for i in brange])
ev.ini_files(brange)

mc_i = 1


for i in np.arange(0,size)[::run_jump]:
    ranger = int(i + run_jump)
    if ranger > size:
        ranger -= ranger - size
    file = open("mc_" + str(mc_i) + ".py", "w")
    b_write = "b = np.arange(" + str(brange[int(i)]) + "," + str(brange[int(ranger)]) + "," + str(b_step) + ") \n"
    #i_start = str(i)
    #i_final = str(i + i_step)
    file.write("import sys \n")
    file.write("sys.path.insert(0,'../') \n")
    file.write("import evac as ev \n")
    file.write("import numpy as np \n")
    #file.write("i_start = " + i_start + " \n")
    #file.write("i_final = " + i_final + " \n")
    #file.write("b = np.array([30]) \n")
    file.write(b_write)
    file.write("b = np.array([round(i,3) for i in b]) \n")
    file.write("ev.main(b) \n")

    file.close()
    mc_i += 1
    
    
