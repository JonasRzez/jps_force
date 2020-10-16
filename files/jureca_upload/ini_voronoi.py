import numpy as np
import os
import pandas as pd
vor_i = 1
test_i = np.arange(0,2)
test_f = np.arange(1,3)
path = pd.read_csv("path.csv")['path'][0]
for i,f in zip(test_i,test_f):

    file = open("vor_" + str(vor_i) + ".py", "w")
    #i_start = str(i)
    #i_final = str(i + i_step)
    
    file.write("import evac as ev \n")
    file.write("import os \n")
    file.write("import pandas as pd \n")
    file.write("import numpy as np \n")
    file.write("import trajectory_vornoi as tv \n")
    file.write("path = pd.read_csv('path.csv')['path'][0] \n")
    file.write("os.system('rm '+ path + 'density/file_list*') \n" )
    voronoi = "tv.voronoi_density" + "(" + str(i) + "," + str(f) + ")"
    file.write(voronoi)


    file.close()
    print(vor_i)
    vor_i += 1
    
    
