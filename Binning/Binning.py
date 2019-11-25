import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Binning:
    '''
    1. Description of the arguments:
       x_col = column to be binned
       y_col = target variable
       bins  = no of bins, default = 6
       plot_type =  1. bar
                    2. scatter
       bin_type(default: 1) =  1 (for equal_data) or 2 (for equal_range)
    '''

    def __init__(self, df):
        self.df = df

    def equal_data(self, x_col, y_col, bins=6):
        '''
        Bins have equal data points
        '''
        if self.df[x_col].dtypes == 'O':
            temp_df = df.loc[:, [x_col, y_col]]
            return temp_df[[x_col, y_col]].groupby(x_col, as_index=False).mean()
        else:
            temp_df = df.loc[:, [x_col, y_col]]
            labels = range(1, bins + 1)
            temp_df[x_col + '_bin'] = pd.qcut(temp_df[x_col], bins, labels=labels)
            temp_df['bin_range'] = pd.qcut(temp_df[x_col], bins)
            count = temp_df[x_col + '_bin'].value_counts(sort=False).values
            df1 = temp_df[[x_col + '_bin', y_col]].groupby([x_col + '_bin'], as_index=False).mean()
            df2 = temp_df[['bin_range', y_col]].groupby(['bin_range'], as_index=False).mean()
            df1 = df1.merge(df2, how='left')
            df1['count'] = count
            return df1.T.reindex([x_col + '_bin', 'bin_range', 'count', y_col]).T

    def equal_range(self, x_col, y_col, bins=6):
        '''
        Equal range bins
        '''
        if self.df[x_col].dtypes == 'O':
            temp_df = df.loc[:, [x_col, y_col]]
            return temp_df[[x_col, y_col]].groupby(x_col, as_index=False).mean()
        else:
            temp_df = df.loc[:, [x_col, y_col]]
            labels = range(1, bins + 1)
            temp_df[x_col + '_bin'] = pd.cut(temp_df[x_col], bins, labels=labels)
            temp_df['bin_range'] = pd.cut(temp_df[x_col], bins)
            count = temp_df[x_col + '_bin'].value_counts(sort=False).values
            df1 = temp_df[[x_col + '_bin', y_col]].groupby([x_col + '_bin'], as_index=False).mean()
            df2 = temp_df[['bin_range', y_col]].groupby(['bin_range'], as_index=False).mean()
            df1 = df1.merge(df2, how='left')
            df1['count'] = count
            return df1.T.reindex([x_col + '_bin', 'bin_range', 'count', y_col]).T

    def plot(self, x_col, y_col, bins=6, plot_type='bar', bin_type=1):
        '''
        If type 1 then equal_data points graph, if type 2 then equal_range graph. By default type = 1
        '''
        plt.figure(figsize=(12, 6))
        sns.set_style('darkgrid')
        if bin_type == 1:
            temp_df = self.df.loc[:, [x_col, y_col]]
            if self.df[x_col].dtypes == 'O':
                pass
            else:
                labels = range(1, bins + 1)
                temp_df[x_col] = pd.qcut(temp_df[x_col], bins, labels=labels)
                temp_df = temp_df[[x_col, y_col]].groupby(x_col, as_index=False).mean()
            if plot_type == 'bar':
                sns.barplot(x=x_col, y=y_col, data=temp_df, palette="Blues_d", ci=0)
            elif plot_type == 'scatter' and self.df[x_col].dtypes != 'O':
                sns.regplot(temp_df[x_col].astype(int), temp_df[y_col].astype(int), ci=None)
            else:
                print('For scatter plot data type of x_col should be continuous')
            plt.xlabel(x_col, fontsize=14)
            plt.ylabel('Avg ' + y_col, fontsize=14)
        elif bin_type == 2:
            temp_df = self.df.loc[:, [x_col, y_col]]
            if self.df[x_col].dtypes == 'O':
                temp_df = self.df.loc[:, [x_col, y_col]]
            else:
                temp_df = self.df.loc[:, [x_col, y_col]]
                labels = range(1, bins + 1)
                temp_df[x_col] = pd.cut(temp_df[x_col], bins, labels=labels)
                temp_df = temp_df[[x_col, y_col]].groupby(x_col, as_index=False).mean()
            if plot_type == 'bar':
                sns.barplot(x=x_col, y=y_col, data=temp_df, palette="Blues_d", ci=0)
            elif plot_type == 'scatter' and self.df[x_col].dtypes != 'O':
                sns.regplot(temp_df[x_col].astype(int), temp_df[y_col].astype(int), ci=None)
            else:
                print('For scatter plot data type of x_col should be continuous')
            plt.xlabel(x_col, fontsize=14)
            plt.ylabel('Avg ' + y_col, fontsize=14)