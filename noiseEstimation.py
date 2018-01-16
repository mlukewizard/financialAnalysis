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

ArrayUnit = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
F = np.zeros([3*len(coinList), 3*len(coinList)])
for i in range(len(coinList)):
    F[0+3*i:0+3*(i+1),0+3*i:0+3*(i+1)] = deepcopy(ArrayUnit)

#Gets all the coin price data
AllCoins = getCoinData(coinList, numDays)

#Gets the line of best fit (non causally)
AllCoinsNoNoise = np.zeros([len(coinList), numDays])
filterLength = 11
for i in range(len(coinList)):
    AllCoinsNoNoise[i,:] = np.convolve(AllCoins[i,:], (np.ones(filterLength)/filterLength))[int(np.floor(filterLength/2)):-int(np.floor(filterLength/2))]

#Extracts the noise from the signal
AllCoinsNoise = AllCoins - AllCoinsNoNoise

AllCoinsNoNoiseGradients = np.zeros([len(coinList), numDays])
for i in range(len(coinList)):
    for j in range(1, numDays):
        AllCoinsNoNoiseGradients[i,j] = AllCoinsNoNoise[i,j] - AllCoinsNoNoise[i,j-1]
AllCoinsNoNoiseGradientGradients = np.zeros([len(coinList), numDays])
for i in range(len(coinList)):
    for j in range(2, numDays):
        AllCoinsNoNoiseGradientGradients[i,j] = AllCoinsNoNoiseGradients[i,j] - AllCoinsNoNoiseGradients[i,j-1]
AllCoinsModelPredicted = np.zeros([3 * len(coinList), numDays + 1])

#You need to track the difference between your model and the NoNoise curve. You also need to figure out
#how to update the the acceleration in the model
state = np.zeros([3*len(coinList)])
for i in range(2, numDays):
    for j in range(len(coinList)):
        state[3*j+0] = AllCoins[j, i]
        state[3*j+1] = AllCoinsNoNoiseGradients[j, i]
        state[3*j+2] = AllCoinsNoNoiseGradientGradients[j, i]
    AllCoinsModelPredicted[:, i] = np.dot(F, state)
'''
#Generating a new series where the end of it is predicted points
predictedPoints = deepcopy(AllCoins)
predictedPoints[:,-trainingRange:] = 0
for i in range(trainingRange):
    AllCoinsRestricted = AllCoins[:,0:-trainingRange+i]
    predictedPoints[:, -trainingRange+i] = generatePredictedPoint(AllCoinsRestricted, lookback)
#You cant access the end of the array with AllCoins[:,0:-trainingRange+i] so add on the last one
extendedPointSet = np.zeros([len(coinList), numDays + 1])
extendedPointSet[:,0:-1] = predictedPoints
extendedPointSet[:,-1] = generatePredictedPoint(AllCoins, lookback)
'''

plt.plot(AllCoinsNoise[0,:])
plt.plot(AllCoinsModelPredicted[0, :])
#plt.scatter(extendedXArray[0,:], extendedPointSet[0,:], color = 'r')
plt.show()
print('Script finished')