import evac as ev
import numpy as np
import pandas as pd
b = np.load("ini_b.npy")
b = np.array([round(i,3) for i in b])
ev.main(b)
