#smoothing
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy import stats

plt.figure(figsize=(25, 3.5))
rads=pd.DataFrame(rads)
rads.columns=['radius']
# plt.plot(rads['radius'].rolling(window=5, min_periods=1).mean(), alpha =3, c= 'red')
plt.plot(rads['radius'].rolling(window=8, min_periods=1).mean(), alpha =3, c= 'green')    
# plt.plot(rads['radius'].rolling(window=20, min_periods=1).mean(), alpha =3, c= 'blue')
plt.show()
