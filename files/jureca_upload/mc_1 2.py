import sys 
sys.path.insert(0,'../') 
import evac as ev 
import numpy as np 
b = np.arange(1.2,1.3,0.01) 
b = np.array([round(i,3) for i in b]) 
ev.main(b) 
