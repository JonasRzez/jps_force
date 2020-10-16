import evac as ev 
import os 
import pandas as pd 
import numpy as np 
import trajectory_vornoi as tv 
path = pd.read_csv('path.csv')['path'][0] 
os.system('rm '+ path + 'density/file_list*') 
tv.voronoi_density(0,1)