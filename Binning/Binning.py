import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Binning:

    '''
    1. Description of the arguments:
       x_col = column to be binned
       y_col = target variable
       bins  = no of bins, default = 6
    '''

    def __init__(self, df, x_col, y_col, bins = 6):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.bins = bins

    def equal_data(self):
        '''
        Each bin have equal data points
        '''
        temp_df = self.df.loc[:, [self.x_col, self.y_col]]
        labels = range(1, self.bins + 1)
        temp_df[self.x_col + '_bin'] = pd.qcut(temp_df[self.x_col], self.bins, labels = labels)
        return temp_df[[self.x_col + '_bin', self.y_col]].groupby(self.x_col + '_bin',as_index= False).mean()

    def equal_range(self):
        '''
        Binning based on equal range of values
        '''
        temp_df = self.df.loc[:, [self.x_col, self.y_col]]
        labels = range(1, self.bins + 1)
        temp_df[self.x_col + '_bin'] = pd.cut(temp_df[self.x_col], self.bins, labels=labels)
        return temp_df[[self.x_col + '_bin', self.y_col]].groupby(self.x_col + '_bin', as_index=False).mean()

    def plot(self, type = 1):
        '''
        If type 1 then equal_data points graph, if type 2 then equal_range graph. By default type = 1
        '''
        sns.set_style('darkgrid')
        if type == 1:
            temp_df = self.df.loc[:, [self.x_col, self.y_col]]
            labels = range(1, self.bins + 1)
            temp_df[self.x_col + '_bin'] = pd.qcut(temp_df[self.x_col], self.bins, labels=labels)
            dfn = temp_df[[self.x_col + '_bin', self.y_col]].groupby(self.x_col + '_bin', as_index=False).mean()
            sns.relplot(x= self.x_col + '_bin', y = self.y_col, data = dfn)
            plt.xlabel(self.x_col + '_bin', fontsize = 14)
            plt.ylabel(self.y_col, fontsize = 14)
        elif type == 2:
            temp_df = self.df.loc[:, [self.x_col, self.y_col]]
            labels = range(1, self.bins + 1)
            temp_df[self.x_col + '_bin'] = pd.cut(temp_df[self.x_col], self.bins, labels=labels)
            dfn = temp_df[[self.x_col + '_bin', self.y_col]].groupby(self.x_col + '_bin', as_index=False).mean()
            sns.relplot(x= self.x_col + '_bin', y = self.y_col, data = dfn)
            plt.xlabel(self.x_col + '_bin', fontsize = 14)
            plt.ylabel(self.y_col, fontsize = 14)