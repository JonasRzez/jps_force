import evac as ev
import numpy as np


#b = np.arange(0.8,7.1,0.1)
#b = np.arange(1.2,2.1,0.01)
#b = np.array([1.2,1.4,1.6,1.8,2.0,2.3,2.4,2.6,2.8,3.0,3.2, 3.4,3.8,4.2, 4.5,4.9,5.3,5.6])
#b = np.array([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.3,2.6,3.0,3.4,4.5,5.6])
b = np.array([20])
#b = np.array([1.7,1.2,2.3,3.4,4.5,5.6])
#b = np.arange(1.7,1.9,0.01)
#b = np.array([1.2, 1.5,1.7, 2.0,5.6])
#b = np.array([1.2, 2.3,3.4, 4.5,5.6])

np.save("ini_b.npy", b)
b = np.array([round(i, 3) for i in b])
ev.ini_files(b)
