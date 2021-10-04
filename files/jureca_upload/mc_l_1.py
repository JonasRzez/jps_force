import sys 
sys.path.insert(0,'../') 
import evac_large as ev 
import numpy as np 
i_start = 0 
i_end = 1 
b = np.array([30]) 
b = np.array([round(i,3) for i in b]) 
esigma = np.array([-0.2,-0.15,-0.1,-0.05,-0.0,0.05,0.1,0.15,0.2,0.25])
ev.main(b,i_start,i_end,esigma,i_start) 
