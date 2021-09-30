import sys 
sys.path.insert(0,'../') 
import evac_large as ev 
import numpy as np 
i_start = 0 
i_end = 1 
b = np.array([40]) 
b = np.array([round(i,3) for i in b]) 
esigma = np.array([0.7])
ev.main(b,i_start,i_end,esigma,i_start) 
