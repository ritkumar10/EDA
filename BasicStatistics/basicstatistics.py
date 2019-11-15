import pandas as pd 

class statistics(object):
    def __init__(self, df):
        self.df = df
        self.numerics=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
    # @staticmethod    
    def seperate_df(self,df):
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
        num_stats['kurtosis']=df.kurt()
        num_stats['skewness']=df.skew()
        num_stats['variance']=df.var()
        num_stats['coeef_of_variation']=num_stats['std']/num_stats['mean'] if num_stats.mean else np.nan
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
    def cat_stats(df):
        cat_stats=df.describe().T.reset_index().rename(columns={'index':'Feature'})
        return cat_stats

    def stat(self):
        stats_df=self.create_df(self.df)
        df_num,df_cat=self.seperate_df(self.df)
        stats_num=self.num_stats(df_num)
        stats_cat=self.cat_stats(df_cat)
        stats_df=stats_df.merge(stats_num,how='left',on='Feature')
        stats_df=stats_df.merge(stats_cat,how='left',on='Feature')
        stats_df=stats_df.fillna('-')
        return stats_df



class missing(object):
    def __init__(self,df):
        self.df=df

    def missing(self):
        missing=self.df.apply(lambda x: sum(x.isnull().values), axis = 0)
        return missing

    @staticmethod
    def mean_imputation(df):
        return df.fillna(df.mean())

    @staticmethod
    def median_imputation(df):
        return df.fillna(df.quantile(0.5))

    def num_impute(self,type):
        if type=='mean':
            self.df=self.mean_imputation(self.df)
        if type=='median':
            self.df=self.median_imputation(self.df)
        return self.df
    
    

