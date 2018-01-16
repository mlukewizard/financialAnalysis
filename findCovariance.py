from pypoloniex import LoadPairs, TimeSeries
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.api import VAR, DynamicVAR
from matplotlib import pyplot as plt
import numpy as np
import datetime
from scipy import signal
from tabulate import tabulate
import pandas as pd

#This defines a couple of functions, these are used for filtering out the
#long term fluctuations so you can look at the prices on a short term time
#scale. Look into butterworth high pass filters if youre interested.
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

today = datetime.date.today()
startDay = today - datetime.timedelta(days= -300)

coinList = ['BTC', 'ETH', 'LTC', 'XRP', 'REP', 'ETC', 'STR', 'ZEC', 'XMR', 'NXT', 'DASH']
AllCoins = np.zeros([len(coinList), 335])
AllCoinsHighFreq = np.zeros([len(coinList), 335])

print(coinList)

for i in range(len(coinList)):
    coin = coinList[i]
    print(coin)
    sess = TimeSeries()
    pair = ('USDT', coin)
    #sess.getData(pair, 86400, today.strftime('%d/%m/%Y'), startDay.strftime('%d/%m/%Y'))
    sess.getData(pair, 86400, '01/01/2017', '01/12/2017')
    AllCoins[i,:] = sess.data._series['close'].values/np.max(sess.data._series['close'].values)

#plt.plot(AllCoins[1,:])
#plt.plot(AllCoins[6,:])
#plt.show()

#Prints the covariance matrix
print('Below is the covariance matrix for BTC, ETH and LTC. You can see that all the values of the matrix are positive'
      + ' which basically shows that when 1 goes up, the other goes up.')
AllCoinsCoVar = np.cov(AllCoins)
df = pd.DataFrame(AllCoinsCoVar)/np.max(np.max(AllCoinsCoVar))
df.to_csv("/media/sf_sharedFolder/financePrediction/no_filter.csv")

#Filters for high frequencies
for i in range(len(coinList)):
    AllCoinsHighFreq[i,:] = butter_highpass_filter(AllCoins[i,:], 3, 30)#/np.max(AllCoins[0,:])

#Prints the covariance matrix for high frequencies
HighFreqCoVar = np.cov(AllCoinsHighFreq)
print('Below is the covariance matrix for the high frequency components of BTC, ETH and LTC. Again all the values ' 
 'of the matrix are positive. N.B they are significantly less than the values in the other matrix simply because'
      'Ive normalised this one')
df = pd.DataFrame(HighFreqCoVar)/(np.max(np.max(HighFreqCoVar)))
df.to_csv("/media/sf_sharedFolder/financePrediction/high_freq.csv")

#Plots the graph
#BTCPlot = plt.plot(BTCHighFreq[0:100])
#ETHPlot = plt.plot(ETHHighFreq[0:100])
#LTCPlot = plt.plot(LTCHighFreq[0:100])
#plt.title('Showing the fluctuation of the high frequency components of BTC, ETH and LTC')
#plt.show()