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
import os

def generateMARPrediction(timeSeries, lookbackPeriod):
    model = VAR(zip(*timeSeries))
    results = model.fit(lookbackPeriod)
    predictions = results.forecast(np.transpose(timeSeries[:,-lookbackPeriod:]), 1)
    return np.transpose(predictions)[:,0]

def getCoinData(myCoinList, numberOfDays):
    # This just gets all the data into AllCoins
    folder = '/media/sf_sharedFolder/financePrediction/'
    today = datetime.date.today()
    for fname in os.listdir(folder):
        if fname == 'PriceData' + str(today) + '.npy':
            AllCoinsData = np.load(folder + fname)
            print('Loaded price data from file')
            return AllCoinsData
    try:
        AllCoinsData = np.zeros([len(myCoinList), numberOfDays])
        startDay = today - datetime.timedelta(days=numberOfDays - 1)
        for i in range(len(myCoinList)):
            coin = myCoinList[i]
            print(coin)
            sess = TimeSeries()
            pair = ('USDT', coin)
            sess.getData(pair, 86400, startDay.strftime('%d/%m/%Y'), today.strftime('%d/%m/%Y'))
            data = sess.data._series['close'].values
            AllCoinsData[i, :] = data
        np.save(folder + 'PriceData' + str(today) + '.npy', AllCoinsData)
        print('Saving new price data')
        return AllCoinsData
    except:
        gotFile = False
        for fname in os.listdir(folder):
            if fname.__contains__('PriceData'):
                AllCoinsData = np.load(folder + fname)
                gotFile = True
                print('No recent price data avaiable, and an error getting new price data. Loading data from ' + fname)
                return AllCoinsData
        if gotFile == False:
            raise Exception('No method of getting price data')

def buildXArray(numDays):
    # Building the array of indexes and dates
    today = datetime.date.today()
    startDay = today - datetime.timedelta(days=numDays - 1)
    dateList = []
    for x in range(0, numDays):
        dateList.insert(0, today - datetime.timedelta(days=x))
    return np.vstack([np.linspace(0, numDays - 1, numDays, dtype=int), dateList])