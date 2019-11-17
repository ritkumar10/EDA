import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import Outlier Detection Algorithms
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from sklearn.preprocessing import MinMaxScaler

def univariate_outliers(df, method, x_col, visualize):
    if method in ['3sigma']:
        mu = df[x_col].mean()
        std = df[x_col].std()
        upper_extreme_3sig = mu + 3 * std
        lower_extreme_3sig = mu - 3 * std

        if visualize:
            plt.figure(figsize=(20, 8))
            sns.set_style("darkgrid")
            ax = sns.distplot(df[x_col].dropna())
            plt.title(x_col + ' Distribution', fontsize=20)
            plt.axvline(x=lower_extreme_3sig, linestyle='--', c='r')
            plt.axvline(x=upper_extreme_3sig, linestyle='--', c='r')
            plt.text(lower_extreme_3sig, (ax.get_ylim()[1]) / 2, s='$\\mu$ - 3*$\\sigma$ =  ' +
                                                                   str(np.round(lower_extreme_3sig, 2)), fontsize=22,
                     rotation=90, ha='right', va='center')
            plt.text(upper_extreme_3sig, (ax.get_ylim()[1]) / 2, s='$\\mu$ + 3*$\\sigma$ =  ' +
                                                                   str(np.round(upper_extreme_3sig, 2)), fontsize=22,
                     rotation=90, ha='left', va='center')
            ax.set_xlabel(x_col, fontsize=18)
            plt.show()
        else:
            df['Outliers'] = 0
            df.loc[(df[x_col] < lower_extreme_3sig), 'Outliers'] = 1
            df.loc[(df[x_col] > upper_extreme_3sig), 'Outliers'] = 1

            return df, lower_extreme_3sig, upper_extreme_3sig

    else:
        if visualize:
            plt.figure(figsize=(8, 6))
            sns.set_style("darkgrid")
            ax = sns.boxplot(y=df[x_col].dropna())
            plt.title('Boxplot' + ' Distribution', fontsize=20)
            ax.set_ylabel(x_col, fontsize=16)
            plt.show()
        else:
            _1st_qnt = df[x_col].quantile(0.25)
            _3rd_qnt = df[x_col].quantile(0.75)
            IQR = _3rd_qnt - _1st_qnt

            upper_extreme_boxplt = _3rd_qnt + 1.5 * IQR
            lower_extreme_boxplt = _1st_qnt - 1.5 * IQR

            df['Outliers'] = 0
            df.loc[(df[x_col] < lower_extreme_boxplt), 'Outliers'] = 1
            df.loc[(df[x_col] > upper_extreme_boxplt), 'Outliers'] = 1

            return df, lower_extreme_boxplt, upper_extreme_boxplt

def bivariate_outliers(df, method, x_col, y_col, outliers_fraction, visualize):
    dfx = df.loc[:, [x_col, y_col]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    dfx.loc[:, [x_col, y_col]] = scaler.fit_transform(dfx.loc[:, [x_col, y_col]])

    X1 = dfx[x_col].values.reshape(-1, 1)
    X2 = dfx[y_col].values.reshape(-1, 1)

    X = np.concatenate((X1, X2), axis=1)

    random_state = np.random.RandomState(42)

    classifiers_name = {
        'IForest': 'Isolation Forest',
        'CBLOF': 'Cluster-based Local Outlier Factor (CBLOF)',
        'ABOD': 'Angle-based Outlier Detector (ABOD)',
        'Feature Bagging': 'Feature Bagging',
        'HBOS': 'Histogram-base Outlier Detection (HBOS)',
        'KNN': 'K Nearest Neighbors (KNN)',
        'AvgKNN': 'Average KNN'}

    # Seven outlier detection tools to be used
    classifiers = {
        'Isolation Forest': IForest(behaviour='new', contamination=outliers_fraction, random_state=random_state),
        'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                            random_state=random_state),
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, check_estimator=False,
                                          random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean', contamination=outliers_fraction)}

    clf = classifiers[classifiers_name[method]]
    clf.fit(X)

    # prediction of a dfpoint category outlier or inlier
    y_pred = clf.predict(X)

    if visualize == False:
        df[x_col] = y_pred.tolist()
        return df
    else:
        xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1

        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        plt.figure(figsize=(16, 8))

        # copy of dfframe
        dfx['outlier'] = y_pred.tolist()

        # IX1 - inlier feature 1,  IX2 - inlier feature 2
        IX1 = np.array(dfx[x_col][dfx['outlier'] == 0]).reshape(-1, 1)
        IX2 = np.array(dfx[y_col][dfx['outlier'] == 0]).reshape(-1, 1)

        # OX1 - outlier feature 1, OX2 - outlier feature 2
        OX1 = dfx[x_col][dfx['outlier'] == 1].values.reshape(-1, 1)
        OX2 = dfx[y_col][dfx['outlier'] == 1].values.reshape(-1, 1)

        print('OUTLIERS: ', n_outliers, ',', 'INLIERS: ', n_inliers, ',', 'Detection Method:', classifiers_name[method])

        # threshold value to consider a dfpoint inlier or outlier
        threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)

        # decision function calculates the raw anomaly score for every point
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)

        # fill blue map colormap from minimum anomaly score to threshold value
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)

        # draw red contour line where anomaly score is equal to thresold
        a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

        # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
        plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

        b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')

        c = plt.scatter(OX1, OX2, c='black', s=20, edgecolor='k')

        plt.axis('tight')

        # loc=2 is used for the top left corner
        plt.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'inliers', 'outliers'],
            prop=matplotlib.font_manager.FontProperties(size=16),
            loc='best')

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title(method, fontsize=20)
        plt.xlabel(x_col, fontsize=16)
        plt.ylabel(y_col, fontsize=16)
        plt.show()

def outlier_detection(df, method, x_col, y_col, outliers_fraction, visualize = False):
    if method in ['3sigma', 'boxplot']:
        return univariate_outliers(df, method, x_col, visualize)
    else:
        return bivariate_outliers(df, method, x_col, y_col, outliers_fraction, visualize)

class OutlierDetection:
    '''
    A. This OutlierDetection class has 3 methods (class functions):
       1. Outliers : It performs univariate and bivariate outlier detection based on the arguments like method, x_col and y_col
                     (target variable) passed and returns a dataframe of the column passed with the addition of an 'Outliers'
                     column having 0 and 1 as its value with 1 representing outliers and 0 for normal data points.

       2. OutliersFullData: It also does the same thing as Outliers method but for the complete data. It returns a dataframe with
                            column names same as the columns in the input data but with entries as 0 and 1, where 1 is for outliers.

       3. plot: This function plots the distribution and boxplot distribution in case of method like '3sigma' or 'boxplot' combined
                with a x_col passed. It plots a scatter plot of x_col and y_col showing outliers and normal points in case of
                methods like one of ['IForest', 'CBLOF', 'ABOD', 'Feature Bagging', 'HBOS', 'KNN', 'AvgKNN'] being passed.


    B. Description of the arguments to be passed:
       x_col = predictor
       y_col = target variable to be predicted
       method = for univariate outliers: [1. '3sigma', 2. 'boxplot']
                for bivariate outliers [1. 'IForest', 2. 'CBLOF', 3. 'ABOD', 4. 'Feature Bagging', 5. 'HBOS', 6. 'KNN', 7. 'AvgKNN']
                Note: IForest and CBLOF performs best
       outlier_fraction (default: 0.05) = fraction of outliers that is being expected in the data. Only for bivariate outlier detection

    C. Univariate Outliers Detection
       Performed on single columns using methods like '3sigma' or 'boxplot'.

       1. '3sigma' : Datapoints less than (mean - 3 * std) and greater than  (mean + 3 * std) are considered outliers. Only
                     applicable if the distribution is normal.
       2. 'boxplot': Datapoints less than (1st quantile - 1.5 * IQR) and greater than  (1st quantile + 1.5 * IQR) are
                     considered outliers.

    D. Bivariate Outliers Detection
       Performed on two variables, where x_col is the predictor and y_col is the target variable using methods like:

       1. IForest : Isolation Forest

           A set of trees is used to partition the data and outliers are determined by looking at the partitioning
           and seeing how isolated a leaf is in the overall structure. Isolation Forest handles multidimensional
           data well.

       2. CBLOF: Cluster-based Local Outlier Factor

           The CBLOF calculates the outlier score based on cluster-based local outlier factor. An anomaly score is
           computed by the distance of each instance to its cluster center multiplied by the instances belonging
           to its cluster.

       3. ABOD: Angle-based Outlier Detector

           The method measures the distance of every data point from its neighbors, taking into account the
           distance between those neighbors -- the variance of the cosine scores is the metric used for outlier
           detection. ABOD handles multidimensional data well. PyOD includes 2 versions:

           1) Fast ABOD -- only using the k nearest neighbors, and
           2) Original ABOD -- Taking into account all data points.

       4. Feature Bagging

           A feature bagging detector fits a number of base detectors on various sub-samples of the dataset. It
           uses averaging or other combination methods to improve the prediction accuracy. By default, Local
           Outlier Factor (LOF) is used as the base estimator. However, any estimator could be used as the base
           estimator, such as kNN and ABOD.


       5. HBOS: Histogram-base Outlier Detection

           This is an effective system for handling unsupervised data, it assumes feature independence. The metric
           used for outlier detection is the construction of histograms and measuring distance from the histogram.
           It’s much faster than multivariate approaches, but at the cost of lower accuracy.

       6. KNN: K Nearest Neighbors

           For each data point, the distances from its k nearest neighbors are looked at for outlier detection.
           PyOD supports 3 versions of kNN:
           1) using the distance from the k-th nearest neighbor as the metric for outlier detection (default),
           2) using the average of the k nearest neighbor distances as the metric, and
           3) using the median of the k nearest neighbor distances as the metric.

       7. AvgKNN: Average KNN

           KNN using the average of the k nearest neighbor distances as the metric for outlier detection.
    '''

    def __init__(self, df):
        self.df = df

    def Outliers(self, method, x_col, y_col = None, outlier_fraction = 0.05):

        dfc = self.df.loc[:, self.df.columns]
        if y_col == None:
            dfn, min, max = outlier_detection(dfc, method, x_col, y_col, outlier_fraction)

            print('\n', 'Lower Boundary: {}'.format(min), '\n',
                  'Upper boundary: {}'.format(max))

            dfx = dfn.loc[:, [x_col, 'Outliers']]
            return dfx

        else:
            dfn = outlier_detection(dfc, method, x_col, y_col, outlier_fraction)

            dfx = self.df.loc[:, [y_col, x_col]]
            dfx['Outliers'] = dfn.loc[:, x_col]
            return dfx

    def Outliers_FullData(self, method, y_col = None, outlier_fraction = 0.05):

        dfc = self.df.loc[:, self.df.columns]
        df_output = pd.DataFrame()
        for column in self.df.columns:
            if self.df[column].dtypes != 'O' and column != y_col:

                if y_col == None:
                    dfn, p, q = outlier_detection(dfc, method, column, y_col, outlier_fraction)
                    df_output = pd.concat([df_output, dfn['Outliers']], axis=1)
                    df_output.rename(columns={'Outliers': column}, inplace=True)
                else:
                    dfn = outlier_detection(dfc, method, column, y_col, outlier_fraction)
                    df_output = pd.concat([df_output, dfn[column]], axis=1)
        return df_output

    def plot(self, method, x_col, y_col = None, outlier_fraction = 0.05):
        visualize = True
        outlier_detection(self.df, method, x_col, y_col, outlier_fraction, visualize)

    def __str__(self):
        return '{}'.format(help(OutlierDetection))
