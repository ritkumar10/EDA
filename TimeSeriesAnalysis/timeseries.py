import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

class ts:

    def __init__(self, df, date_col, value_col):
        '''
        Args:

        df: dataframe
        date_col: date time column(datatype should be datetime)
        value_col: timeseries column to be analysed
        '''
        self.df = df
        self.date_col = date_col
        self.value_col = value_col
        self.dfn = (self.df.loc[:, [date_col, value_col]].set_index(date_col))[value_col]

    def plot(self, window=12):
        '''
        Plots two graphs, 1st graph shows the distribution of actual values vs time and 2nd graph shows the actual
        values distribution along with moving average trend and also distribution after removing trend.

        Args:

        window(default = 12): for calculating rolling mean
        '''
        sns.set_style('darkgrid')
        self.dfn.plot(figsize=(20, 10), linewidth=3, fontsize=20, label='Actual values')
        plt.title(self.dfn.name, fontsize=20)
        plt.xlabel('Date', fontsize=20)

        plt.figure(figsize=(20, 10))
        self.dfn.plot(linewidth=3, fontsize=20, label='Actual values')
        self.dfn.rolling(window).mean().plot(linewidth=3, fontsize=20,
                                             label='Moving average trend:\n window size= {}'.format(window))
        self.dfn.diff().plot(linewidth=3, fontsize=20, label='After removing trend')
        plt.xlabel('Date', fontsize=20)
        plt.title(self.dfn.name, fontsize=20)
        plt.legend(loc='best', fontsize=14)

    def isStationary(self):
        '''
        Checks whether the timeseries data is stationary or not by performing Dickey-Fuller test.
        '''
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(self.dfn, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

    def autocorrelationPlot(self):
        '''
        Autocorrelation plot for determining seosonal periodicity i.e. the lag with the highest autocorrelation
        value other than 0
        '''
        sns.set_style('darkgrid')
        plt.figure(figsize=(10, 6), linewidth=2)
        pd.plotting.autocorrelation_plot(self.dfn)
        plt.grid()

    def decomposedPlot(self, period=12):
        '''
        plots the original, trend, seasonal and residual components of the timeseries data

        Args:

        period(default = 12): seasonal component periodicity
        '''
        sns.set_style('darkgrid')
        rcParams['figure.figsize'] = 20, 16
        rcParams['lines.linewidth'] = 3
        decomposed = sm.tsa.seasonal_decompose(self.dfn, freq=period, model="additive")
        decomposed.plot()
        plt.xlabel('Date', fontsize=20)

    def makeStationary(self):
        '''
        Performs first odrer differencing to make the data stationary(have to check by calling 'isStationary'
        method) and returns a dataframe
        '''
        return pd.DataFrame(self.dfn.diff()).reset_index().dropna()