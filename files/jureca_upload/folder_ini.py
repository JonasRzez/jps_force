import evac_large as ev
import numpy as np


#b = np.arange(0.8,7.1,0.1)
#b = np.arange(1.2,2.1,0.01)
#b = np.array([1.2,1.4,1.6,1.8,2.0,2.3,2.4,2.6,2.8,3.0,3.2, 3.4,3.8,4.2, 4.5,4.9,5.3,5.6])
#b = np.array([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.3,2.6,3.0,3.4,4.5,5.6])
b = np.array([50])
#b = np.array([1.7,1.2,2.3,3.4,4.5,5.6])
#b = np.arange(1.7,1.9,0.01)
#b = np.array([1.2, 1.5,1.7, 2.0,5.6])
#b = np.array([1.2, 2.3,3.4, 4.5,5.6])
#esigma = np.arange(0.4,0.55,0.01)
#esigma0 = np.arange(0.2,0.26,0.01)

#esigma1 = np.array([0.2,0.3,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.45,0.46,0.47,0.48,0.49,0.5,0.51])
#esigma = np.append(esigma0,esigma1)
esigma = [0.15,0.3,0.5,0.7,5.0]
esigma = np.array([round(i, 3) for i in esigma])

#esigma = np.array([0.5])
i_start = 0
i_end = 1
ini_i = 0

#np.save("ini_b.npy", b)
b = np.array([round(i, 3) for i in b])
ev.ini_files(b,i_start,i_end,esigma)
#ev.main(b,i_start,i_end,esigma,ini_i)
