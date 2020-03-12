class OutlierImputation:
    '''
    Description of the arguments passed:
    df =  the original dataframe
    col = column in which outliers are to be imputed
    dfn = dataframe returned from OutlierDetection class after running Outlier method
    '''
    def __init__(self,df):
        self.df=df

    def median_imputation(self,col, dfn):
        index = dfn[dfn['Outliers'] == 1].index
        self.df.loc[index, col] = self.df[col].median()
        

    def mean_imputation(self,col, dfn):
        index = dfn[dfn['Outliers'] == 1].index
        self.df.loc[index, col] = self.df[col].mean()
       

    def _3sigma_capping(self,col, dfn):
        mu = self.df[col].mean()
        std = self.df[col].std()
        lowest = mu - 3 * std
        highest = mu + 3 * std

        self.df.loc[self.df[col] > highest, col] = highest
        self.df.loc[self.df[col] < lowest, col] = lowest
      

    def _IQR_capping(self,col, dfn):
        _1st_qnt = self.df[col].quantile(0.25)
        _3rd_qnt = self.df[col].quantile(0.75)
        IQR = _3rd_qnt - _1st_qnt

        upper = _3rd_qnt + 1.5 * IQR
        lower = _1st_qnt - 1.5 * IQR

        self.df.loc[(df[col] < lower), col] = lower
        self.df.loc[(df[col] > upper), col] = upper
      

    def _3IQR_capping(self,col, dfn):
        _1st_qnt = self.df[col].quantile(0.25)
        _3rd_qnt = self.df[col].quantile(0.75)
        IQR = _3rd_qnt - _1st_qnt

        upper = _3rd_qnt + 3 * IQR
        lower = _1st_qnt - 3 * IQR

        self.df.loc[(df[col] < lower), col] = lower
        self.df.loc[(df[col] > upper), col] = upper

