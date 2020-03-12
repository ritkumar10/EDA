
import pandas as pd
from autoimpute.imputations import SingleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from MissingValueAnalysis import knn as knn


class Missing(object):
    """This module performs missing value analysis and imputation in a dataset.
    This module contains one class - Missing. Use this class to perform three different methods.

    1.missing- compute missing values in each column of a DataFrame.
    2.analyse- Analyse y column with x categorical column for missing values in each category
    3.impute- Impute one column at a time of dataframe using various methods like mean,knn,regression techniques.For methods
    of imputation requiring model building you can pass cols to regress on them.

    """
    def __init__(self,df):
        self.df=df
        self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    def missing(self):
        """
        Calling this method will return the count and percentage of missing values in the data

        :return: Returns a dataframe with three columns- feature,missing_count,missing_perc
        """

        missing_count=self.df.apply(lambda x: sum(x.isnull().values), axis = 0).rename('missing_count')
        missing_perc = (self.df.apply(lambda x: sum(x.isnull().values), axis=0).rename('missing_perc')/self.df.shape[0])*100
        missing=pd.concat([missing_count,missing_perc],axis=1)
        return missing

    def analyse(self,x,y):
        """
        This method can be useful in analysis of a column w.r.t an categorical column in missing value analysis.
        For each category in x count of missing values in y are calculated to find correlations and missing patterns


        :param x: Categorical column name has to be passed as a string
        :param y: analysis column name has to be passed as a string
        :return: returns a dataframe with two columns- category name in x and respective missing value count column
        """
        print(' Missing values in '+x+':',sum(self.df[x].isnull().values))
        print(' Missing values in '+y+':',sum(self.df[y].isnull().values))
        x_=self.df[x].isnull()
        y_=self.df[y].isnull()
        x_y=x_ & y_
        print(' Missing values together in ' + x + ' + ' + y + ': ', x_y.values.sum())
        self.df['null'+y]=self.df[y].isnull().astype(int)
        s=self.df.groupby(x)['null'+y].sum()
        self.df.drop(['null'+y],axis=1,inplace=True)
        return s

    def mean_imputation(self,col):
        """
        imputes mean in null values of the given column
        """
        if self.df[col].dtype not in self.numerics:
            print('Mean not applicable for object columns')
            return
        self.df[col].fillna(self.df[col].mean(),inplace=True)

    def median_imputation(self,col):
        """
        imputes median in null values of the given column
        """
        if self.df[col].dtype not in self.numerics:
            print('Median not applicable for object columns')
            return
        self.df[col].fillna(self.df[col].quantile(0.5),inplace=True)

    def mode_imputation(self,col):
        """
        imputes mode in null values of the given column
        """
        if self.df[col].dtype in self.numerics:
            print('warning: Mode imputed for numeric column')
        self.df[col].fillna(self.df[col].mode()[0],inplace=True)

    def model(self,method,col,predictors):
        """
        imputes the given column with any of the regression techniques.
        """
        if predictors:
            params=predictors
        else:
            params=self.df.select_dtypes(include=self.numerics).columns.to_list()
            if col in params: params.remove(col)
            print(params)
        imputer = SingleImputer(strategy={col:method},predictors={col: params})
        self.df=imputer.fit_transform(self.df)

    def knn_imputation(self,col,predictors,n,n_threshold):
        """
        imputes the given column using knn algorithm
        """
        if predictors:
            params = predictors
        else:
            params = self.df.select_dtypes(include=self.numerics).columns.to_list()
        self.df[col]=knn.knn_impute(self.df[col], self.df[params], n, aggregation_method="mean", numeric_distance="euclidean",
               categorical_distance="jaccard", missing_neighbors_threshold=n_threshold)


    def impute(self,method,col,predictors=None,n=None,n_threshold=None):
        """
        This method is used for imputing columns with missing values, column name and method need to be passed
        For mean,median,mode only col name has to be passed, for regression techqniques columns to be regressed on has to be passed,
        for knn columns to be modelled on, no of neighbours and neighborhood threshold has to be passed.
        It's not necessary the regression models are converged. A warning is raised if max iterations are met
        It's not necessary for knn algo to impute all the missing values.

        :param method: string of method neame to be passed.Possible methods- (mean,median,mode,knn,least squares,binary logistic,multinomial logistic,pmm)
        :param col: string of column name to be imputed
        :param predictors: array of strings of column names to be passed on which modelling methods to be used. Not applicable for mean, median,mode
        :param n: No of neighbours to be accounted for. Only for knn
        :param n_threshold: neighbourhood threshold to impute if only no of missing values in neighbourhood is less than n_threshold * n
        :return:nothing
        """
        if method=='mean' :
            self.mean_imputation(col)
        if method=='median':
            self.median_imputation(col)
        if method=='mode':
            self.mode_imputation(col)
        if method=='knn':
            self.knn_imputation(col,predictors,n,n_threshold)
        else:
            self.model(method,col,predictors)

