import sys 
sys.path.insert(0,'../') 
import evac_large as ev 
import numpy as np 
i_start = 8 
i_end = 9 
b = np.array([50]) 
b = np.array([round(i,3) for i in b]) 
esigma = np.array([0.2,0.4,0.5,1.5])
ev.main(b,i_start,i_end,esigma) 
