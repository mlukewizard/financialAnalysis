from pypoloniex import LoadPairs, TimeSeries
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.api import VAR, DynamicVAR
from matplotlib import pyplot as plt
import numpy as np
import datetime
from scipy import signal
from tabulate import tabulate
import pandas as pd
from copy import deepcopy
from financeFunctions import *

coinList = ['BTC', 'ETH', 'LTC', 'XRP', 'REP', 'ETC', 'STR', 'ZEC', 'XMR', 'NXT', 'DASH']
lookback = 100
numDays = 300
trainingRange = 20

#Builds arrays for plotting and defining time series
xArray = buildXArray(numDays)
extendedXArray = buildXArray(numDays+1)

#Gets all the coin price data
AllCoins = getCoinData(coinList, numDays)

#Generating a new series where the end of it is predicted points
predictedPoints = deepcopy(AllCoins)
predictedPoints[:,-trainingRange:] = 0
for i in range(trainingRange):
    AllCoinsRestricted = AllCoins[:,0:-trainingRange+i]
    predictedPoints[:, -trainingRange+i] = generateMARPrediction(AllCoinsRestricted, lookback)
#You cant access the end of the array with AllCoins[:,0:-trainingRange+i] so add on the last one
extendedPointSet = np.zeros([len(coinList), numDays + 1])
extendedPointSet[:,0:-1] = predictedPoints
extendedPointSet[:,-1] = generateMARPrediction(AllCoins, lookback)

plt.plot(AllCoins[0,:])
plt.scatter(extendedXArray[0,:], extendedPointSet[0,:], color = 'r')
plt.show()
print('Script finished')