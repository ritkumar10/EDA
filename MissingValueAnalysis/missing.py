
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