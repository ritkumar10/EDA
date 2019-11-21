class OutlierImputation:
    '''
    Description of the arguments passed:
    df =  the original dataframe
    col = column in which outliers are to be imputed
    dfn = dataframe returned from OutlierDetection class after running Outlier method
    '''
    def __init__(self):
        pass

    def median_imputation(self, df, col, dfn):
        index = dfn[dfn['Outliers'] == 1].index
        df.loc[index, col] = df[col].median()
        return df

    def mean_imputation(self, df, col, dfn):
        index = dfn[dfn['Outliers'] == 1].index
        df.loc[index, col] = df[col].mean()
        return df

    def _3sigma_capping(self, df, col, dfn):
        mu = df[col].mean()
        std = df[col].std()
        lowest = mu - 3 * std
        highest = mu + 3 * std

        df.loc[df[col] > highest, col] = highest
        df.loc[df[col] < lowest, col] = lowest
        return df

    def _IQR_capping(self, df, col, dfn):
        _1st_qnt = df[col].quantile(0.25)
        _3rd_qnt = df[col].quantile(0.75)
        IQR = _3rd_qnt - _1st_qnt

        upper = _3rd_qnt + 1.5 * IQR
        lower = _1st_qnt - 1.5 * IQR

        df.loc[(df[col] < lower), col] = lower
        df.loc[(df[col] > upper), col] = upper
        return df

    def _3IQR_capping(self, df, col, dfn):
        _1st_qnt = df[col].quantile(0.25)
        _3rd_qnt = df[col].quantile(0.75)
        IQR = _3rd_qnt - _1st_qnt

        upper = _3rd_qnt + 3 * IQR
        lower = _1st_qnt - 3 * IQR

        df.loc[(df[col] < lower), col] = lower
        df.loc[(df[col] > upper), col] = upper
        return df