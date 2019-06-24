# 14 June 2016, @kirkbyl

import numpy as np
import matplotlib.pylab as plt
import collections
import time
import random
import sklearn
from sklearn import linear_model, feature_selection, ensemble

from ecogtools.analysis import IMScorr as IMSmodule
from ecogtools.visualization import plotcorrelations


def extract_ICN_features(patient, proj, projtAxis, metric='IMS', deltaT=-300, interval=600, window=60, calcVar=True, plotCorr=False, shuffle=False):
    """
    Extract ICN features around each time point of interest -- eg IMS scores or pain scores (generalized version of extract_IMS_ICN_features function below)

    Inputs:
        - patient: object of Patient class
        - proj: nICN x tTimepoint matrix of coh projections onto each ICN
        - projtAxis: corresponding time axis in UTC
        - metric: can be 'IMS' or 'pain'

        - deltaT: time in sec before each IMS point
        - interval: time in sec over which to calculate ICN feature (starting from tIMS-deltaT)
        - window: time in sec over which to calculate rolling variance (default 600s or 60 data points if coh measured in 10s bins)
        - calcVar: calculate variance of feature if True
        - plotCorr: plot scatter plots of IMS vs. feature if True
    """

    nICN = proj.shape[0]

    if metric == 'IMS':
        scorePts = np.array(patient.IMS['IMS'])
        scoreTimes = np.array([time.mktime(tt.timetuple())
                               for tt in patient.IMS['IMStimes']])
    elif metric == 'pain':
        scorePts = np.array(patient.painScores['painScore'])
        scoreTimes = np.array([time.mktime(tt.timetuple())
                               for tt in patient.painScores['painTimes']])
    else:
        print 'Unknown metric'

    for ind in range(nICN):

        if calcVar == True:

            feature, featuretAxis = IMSmodule.calculate_coh_variance(
                proj[ind], projtAxis, window=window)

        else:
            feature = proj[ind]
            featuretAxis = projtAxis

        # Add in shuffling of time series for model validation
        if shuffle == True:
            random.shuffle(feature)

        scoreDataframe = IMSmodule.score_vs_feature(
            scorePts, scoreTimes, feature, featuretAxis, deltaT=deltaT, interval=interval)

        xFeature = np.array(scoreDataframe['meanVal'])
        yScore = np.array(scoreDataframe['score'])

        if ind == 0:
            nScore = len(yScore)
            featureMatrix = np.zeros((nScore, nICN))

        featureMatrix[:, ind] = xFeature

        if plotCorr == True:
            fig, ax = plt.subplots(figsize=(8, 8))
            plotcorrelations.scatter_regplot(xFeature, yScore, ax=ax)
            slope, intercept, r, p, stderr = plotcorrelations.scatter_linregress(
                xFeature, yScore, ax=ax)
            ax.set_xlabel('ICN projection, variance = {}'.format(str(calcVar)))
            ax.set_ylabel('Score ({0})'.format(metric))
            ax.set_title('Patient: '+patient.ID+', ICN '+str(ind+1))
            plt.show()

    return featureMatrix, yScore


def ICN_IMS_linmod(featureMatrix, IMS, regType='ElasticNet', crossVal=True, showCorrPlots=False, showErrorPlot=True, normalize=False, l1_ratio_CV=[0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], l1_ratio=0.5, alpha=0.1):
    """    
    - regType: regularization technique. Can be 'Lasso', 'Ridge', 'ElasticNet', or None. If None, standard LinearRegression is used (no regularization)
    """

    X = np.atleast_2d(featureMatrix).T
    y = IMS

    nIMS = len(IMS)

    if regType is not None:

        if crossVal == True:
            if regType == 'Lasso':
                linModel = linear_model.LassoCV(normalize=normalize, cv=nIMS)
            elif regType == 'Ridge':
                linModel = linear_model.RidgeCV(normalize=normalize, cv=nIMS)
            elif regType == 'ElasticNet':
                linModel = linear_model.ElasticNetCV(
                    normalize=normalize, l1_ratio=l1_ratio_CV, cv=nIMS)
            else:
                print('Unknown regularization type')

        elif crossVal == False:
            if regType == 'Lasso':
                linModel = linear_model.Lasso(normalize=normalize, alpha=alpha)
            elif regType == 'Ridge':
                linModel = linear_model.Ridge(normalize=normalize, alpha=alpha)
            elif regType == 'ElasticNet':
                linModel = linear_model.ElasticNet(
                    normalize=normalize, l1_ratio=l1_ratio, alpha=alpha)
            else:
                print('Unknown regularization type')
    else:
        linModel = linear_model.LinearRegression(normalize=normalize)

    linModel.fit(X.T, y)

    return linModel
