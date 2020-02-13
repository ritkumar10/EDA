import pandas as pd
import datetime
import numpy as np

class statistics(object):
    """
    This module gives various kinds of statistics for different columns of the dataframe.
    Stat method gives summary statistics of the pandas dataframe.
    seperate_df method splits dataframe into numerical and object columns.
    combine method can be used to combine two mutually exclusive columns into one.
    Requires: a pandas dataframe for initialisation.
    """
    def __init__(self, df):
        self.df = df
        self.numerics=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
    # @staticmethod    
    def seperate_df(self,df):
        """
        Seperates given dataframe into two : one with numerical columns and other with object type columns
        :param df: Pandas dataframe to be divided
        :return: numerical and object type seperated pandas dataframes
        """
        num_columns=df.select_dtypes(include=self.numerics).columns
        cat_columns=df.select_dtypes(exclude=self.numerics).columns
        df_num=df[num_columns]
        df_cat=df[cat_columns]
        return df_num,df_cat

    # @staticmethod
    def create_df(self,df):
        col_names=df.select_dtypes(include=self.numerics).columns.tolist()+df.select_dtypes(exclude=self.numerics).columns.tolist()
        new_df=df=pd.DataFrame(col_names,columns=['Feature'])
        return new_df

    @staticmethod
    def num_stats(df):
        num_stats=df.describe().T
        num_stats['median']=df.quantile(0.5)
        num_stats['mode']=df.mode().iloc[0]
        num_stats['kurtosis']=df.kurt()
        num_stats['skewness']=df.skew()
        num_stats['variance']=df.var()
        num_stats['coef_of_variation']=num_stats['std']/num_stats['mean'] if num_stats.mean else np.nan
        num_stats['mean_absolute_deviation']=df.mad()
        num_stats['iqr']=num_stats['75%']-num_stats['25%']
        num_stats['1%']=df.quantile(0.01)
        num_stats['99%']=df.quantile(0.99)
        num_stats['5%']=df.quantile(0.05)
        num_stats['95%']=df.quantile(0.95)
        num_stats['outliers']=((df < (num_stats['25%'] - 1.5 * num_stats['iqr'])) | (df > (num_stats['75%'] + 1.5 * num_stats['iqr']))).sum()
        num_stats['outliers%']=num_stats['outliers']/df.shape[0]
        num_stats['unique_num']=df.nunique()
        num_stats['sum']=df.sum()
        num_stats['num_zeros']=(df == 0).astype(int).sum(axis=0)
        num_stats=num_stats.reset_index().rename(columns={'index':'Feature'})
        return num_stats

    @staticmethod
    def freq(df, f_type):
        if f_type == 'min':
            s1 = df.apply(lambda x: x.value_counts().min() if (x.nunique() > 1) else None, axis=0).rename('min_freq')
            s2 = df.apply(lambda x: x.value_counts().reset_index()['index'].iloc[-1] if (x.nunique() > 1) else None, axis=0).rename('min_freq_category')
            s=pd.concat([s1,s2],axis=1)
        if f_type == 'second':
            s1 = df.apply(lambda x: x.value_counts()[1] if (x.nunique() > 1) else None, axis=0).rename('2nd_freq')
            s2 = df.apply(lambda x: x.value_counts().reset_index()['index'].iloc[1] if (x.nunique() > 1) else None, axis=0).rename('2nd_freq_category')
            s = pd.concat([s1, s2], axis=1)
        return s

    def cat_stats(self,df):
        cat_stats=df.describe().T
        cat_stats=cat_stats.rename(columns={'top':'top_freq_category','freq':'top_freq'})
        cat_stats = cat_stats.merge(self.freq(df, 'second'), left_index=True, right_index=True, how='left')
        cat_stats = cat_stats.merge(self.freq(df, 'min'), left_index=True, right_index=True, how='left')
        cat_stats['top_freq/2nd_freq'] = cat_stats['top_freq'] / cat_stats['2nd_freq']
        cat_stats['top_freq/min_freq'] = cat_stats['top_freq'] / cat_stats['min_freq']
        cat_stats = cat_stats.reset_index().rename(columns={'index': 'Feature'})
        return cat_stats

    @staticmethod
    def combine(df,a,b,name):
        """
        Merges two same columns which are mutually exclusive into one
        :param df: dataframe containing the two columns
        :param a: first col name
        :param b: second col name
        :param name: new name for the column
        :return: pandas dataframe with merged columns into one with new name.
        """
        df[a].fillna(0, inplace=True)
        df[b].fillna(0, inplace=True)
        df[name]=df[a]+df[b]
        df.drop([a,b],inplace=True, axis=1)
        return df

    @staticmethod
    def missing(df):

        """
        Calling this method will return the count and percentage of missing values in the data

        :return: Returns a dataframe with three columns- feature,missing_count,missing_perc
        """
        missing_count = df.apply(lambda x: sum(x.isnull().values), axis=0).rename('missing_count')
        missing_perc = df.apply(lambda x: sum(x.isnull().values), axis=0).rename('missing_perc') / df.shape[0]
        missing = pd.concat([missing_count, missing_perc], axis=1)
        return missing.reset_index().rename(columns={'index':'Feature'})

    @staticmethod
    def dtypes(df):

        """
        Calling this method will return the count and percentage of missing values in the data

        :return: Returns a dataframe with three columns- feature,missing_count,missing_perc
        """

        return pd.DataFrame(df.dtypes).reset_index().rename(columns={'index':'Feature',0:'dtype'})

    @staticmethod
    def reorganize(df):
        common_stats=['Feature','dtype','count','unique_count','missing_count','missing_perc']
        num_stats=['min','25%','50%','75%','max','mean','std','median','mode','kurtosis','skewness','variance','coef_of_variation','mean_absolute_deviation','iqr']
        outlier_stats=['1%','99%','5%','95%','outliers','outliers%','num_zeros','sum']
        cat_stats=['top_freq_category','top_freq','2nd_freq_category','2nd_freq','min_freq_category','min_freq','top_freq/2nd_freq','top_freq/min_freq']
        cols=common_stats+num_stats+outlier_stats+cat_stats
        return df[cols]

    def stat(self):
        """
        Following stats are returned for each column :
        For numerical columns:
                (mean, standard deviation,kurtosis,skewness,variance,coefficient of variation,mean absolute deviation,interquartile range),
                (min,25 %ile,50 %ile ,75 %ile,max), (outliers,outlier percentage,1 %ile,5 %ile,99 %ile,95 %ile), (sum, count of zeros in column),
        and for object columns:
                (Mode,frequency of mode,second mode frequency, min frequency of category and their corresponding ratios to mode frequency)
        and count of each column and no of unique values in each column.

        :return: Returns a pandas dataframe of summary statistics with column name as index

        """
        stats_df=self.create_df(self.df)
        df_num,df_cat=self.seperate_df(self.df)
        stats_num=self.num_stats(df_num)
        stats_cat=self.cat_stats(df_cat)
        stats_df=stats_df.merge(stats_num,how='left',on='Feature')
        stats_df=stats_df.merge(stats_cat,how='left',on='Feature')
        stats_df=self.combine(stats_df,'count_x','count_y','count')
        stats_df = self.combine(stats_df, 'unique_num', 'unique', 'unique_count')
        stats_df=stats_df.merge(self.missing(self.df),how='left',on='Feature')
        stats_df = stats_df.merge(self.dtypes(self.df), how='left', on='Feature')
        stats_df=stats_df.fillna('-')
        stats_df=self.reorganize(stats_df)
        return stats_df



    
    

