import pandas as pd
import numpy as np
from matplotlin import pyplot as plt
import glob
from scipy import stats

file_list = sorted(glob.glob('./data/data_frame/*.csv'))

for i in range(10):
    print('sample {}'.format(i+1))
    df = pd.read_csv('./data/data_frame/*.csv'.format(i+1))
    state = df['label'].values
    redious = df['radious'].values
    
    plt.figure(figsize=(20, 5))
    plt.scatter(range(len(radious)), raious, c = state)
    plt.plot(radious, alpha=0.3)
    plt.plot(df['radious'].rolling(window=5, min_periods=1).mean(), alpha =3, c= 'red')
    plt.plot(df['radious'].rolling(window=10, min_periods=1).mean(), alpha =3, c= 'green')    
    plt.plot(df['radious'].rolling(window=20, min_periods=1).mean(), alpha =3, c= 'blue')
    plt.xlim(0,200)
    plt.show()

 
for i in range(len(file_list)):
    print('sample {}'.format(i+1))
    df = pd.read_csv('./data/data_frame/*.csv'.format(i+1))
    state = df['label'].values
    redious = df['radious'].values
    
    plt.figure(figsize=(20, 5))
    plt.scatter(range(len(radious)), raious, c = state)
    plt.plot(radious, alpha=0.3)
    plt.plot(df['radious'].rolling(window=5, min_periods=1).mean(), alpha =3, c= 'red')
    plt.plot(df['radious'].rolling(window=10, min_periods=1).mean(), alpha =3, c= 'green')    
    plt.plot(df['radious'].rolling(window=20, min_periods=1).mean(), alpha =3, c= 'blue')
    plt.xlim(0,200)
    plt.show()
