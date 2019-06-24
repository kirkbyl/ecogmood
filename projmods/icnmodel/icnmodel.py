# @kirkbyl, 7 July 2016
"""
Class for running linear models of ICN projections against IMS: what networks are best predictors of IMS
"""

from scipy.stats import linregress
from scipy.odr import Model, Data, ODR
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
from scipy import odr
import time
import datetime

from ecogtools.analysis import models as linmod, IMScorr as IMSmodule
from ecogtools.tools import utilities as utils
from ecogtools.visualization import plotcorrelations, plotcoherence
from ecogtools.recordingparams import subjects as subj

from projmods import icnetworks


class LinModelICN(object):

    def __init__(self, patient):

        self.patientID = patient.getID()
        self.patient = patient

        self.allBands = ['theta', 'alpha', 'beta', 'gamma']

        return

    def loadICN(self, band):

        self.ICNs = []
        self.ICNprojs = []
        self.ICNtaxis = []
        self.band = band

        if band == 'all':
            for bb, bnd in enumerate(self.allBands):
                net = icnetworks.IntrinsicNetwork(self.patient, bnd)
                if bb == 0:
                    self.ICNs = net.ICN
                    self.ICNprojs = net.proj
                else:
                    self.ICNs = np.dstack((self.ICNs, net.ICN))
                    self.ICNprojs = np.vstack((self.ICNprojs, net.proj))
            self.ICNtaxis = net.tAxis['UTC']

        else:
            net = icnetworks.IntrinsicNetwork(self.patient, band)
            self.ICNs = net.ICN
            self.ICNprojs = net.proj
            self.ICNtaxis = net.tAxis['UTC']

        return

    def getFeatures(self, metric='IMS', calcVar=True, deltaT=-600, interval=1200, window=60, plotCorr=False, band=None, shuffle=False, **kwargs):
        """
        Extract ICN features around each score (eg IMS) time point

        Inputs:
            - metric: 'IMS'
            - calcVar: calculate variance of feature if True        
            - deltaT: time in sec before each IMS point
            - interval: time in sec over which to calculate ICN feature (starting from tIMS-deltaT)
            - window: time in sec over which to calculate rolling variance (default 600s or 60 data points if coh measured in 10s bins)
            - plotCorr: plot scatter plots of score vs. feature if True
            - band: frequency band of ICNs
        """

        if band is not None:
            self.band = band
        else:
            self.band = raw_input(
                "Enter band of interest. Choose from {0} or 'all': ".format(str(self.allBands)))

        self.loadICN(self.band)

        if kwargs:
            if 'calcVar' in kwargs:
                calcVar = kwargs['calcVar']
            if 'deltaT' in kwargs:
                deltaT = kwargs['deltaT']
            if 'interval' in kwargs:
                interval = kwargs['interval']
            if 'window' in kwargs:
                window = kwargs['window']
            if 'shuffle' in kwargs:
                shuffle = kwargs['shuffle']

        featureMatrix, score = linmod.extract_ICN_features(self.patient, self.ICNprojs, self.ICNtaxis, metric=metric,
                                                           deltaT=deltaT, interval=interval, window=window, calcVar=calcVar, plotCorr=plotCorr, shuffle=shuffle)

        self.featureMatrix = featureMatrix
        self.score = score

        self.modelIn = {}
        for modIn in ['calcVar', 'deltaT', 'interval', 'window', 'metric', 'shuffle']:
            self.modelIn[modIn] = eval(modIn)

        return

    def runModel(self, metric='IMS', regType='ElasticNet', crossVal=True, band=None, normalize=False, plotCoeffs=True, alpha=0.01, l1_ratio=0.5, l1_ratio_CV=[0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], **kwargs):
        """
        - regType: regularization technique. Can be 'Lasso', 'Ridge', 'ElasticNet', or None. If None, standard LinearRegression is used (no regularization)
       """

        self.getFeatures(band=band, metric=metric, **kwargs)

        # Run model:
        for modIn in ['crossVal', 'regType']:
            self.modelIn[modIn] = eval(modIn)

        self.modelOut = linmod.ICN_IMS_linmod(self.featureMatrix, self.score, regType=regType,
                                              normalize=normalize, l1_ratio=l1_ratio, l1_ratio_CV=l1_ratio_CV, alpha=alpha, crossVal=crossVal)

        # Training prediction:
        scorePredict_train = np.zeros(len(self.score))

        for tt, testFeature in enumerate(self.featureMatrix):
            scorePredict_train[tt] = self.predictFeature(testFeature)
        self.scorePredict_train = scorePredict_train

        # Leave one out cross-validation:
        scorePredict_test = np.zeros(len(self.score))
        bestICN_LOO = np.zeros(len(self.score))
        score_LOO = np.zeros(len(self.score))
        coeffs_LOO = np.zeros((len(self.score), self.featureMatrix.shape[-1]))

        for ii, scoreTest in enumerate(self.score):

            testFeature = self.featureMatrix[ii].reshape(1, -1)

            y_model = np.delete(self.score, ii)
            X_model = np.delete(self.featureMatrix, ii, axis=0)
            testModel = linmod.ICN_IMS_linmod(X_model, y_model, regType=regType,
                                              normalize=normalize, l1_ratio=l1_ratio, l1_ratio_CV=l1_ratio_CV, alpha=alpha, crossVal=crossVal)

            scorePredict_test[ii] = testModel.predict(testFeature)
            bestICN_LOO[ii] = int(np.argmax(abs(testModel.coef_))+1)
            score_LOO[ii] = testModel.score(self.featureMatrix, self.score)
            coeffs_LOO[ii, :] = testModel.coef_

        self.scorePredict_test = scorePredict_test
        self.bestICN_LOO = bestICN_LOO
        self.score_LOO = score_LOO
        self.coeffs_LOO = coeffs_LOO

        # Make plots:
        if plotCoeffs == True:
            self.plotCoeffs()

        return

    def predictFeature(self, testFeature):

        if len(testFeature.shape) < 2:
            testFeature = testFeature.reshape(1, -1)

        return self.modelOut.predict(testFeature)

    def plotCoeffs(self):

        if not hasattr(self, 'modelOut'):
            print('Running linear model with default parameters')
            self.runModel()

        nFeatures = len(self.modelOut.coef_)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        ax[0].plot(range(1, nFeatures+1), self.modelOut.coef_, 'ko')
        ax[0].axhline(0, c='r', linestyle='--')
        ax[0].set_xlabel('ICN number')
        ax[0].set_ylabel('Coefficient')
        bestICN = np.argmax(abs(self.modelOut.coef_))
        plotcoherence.plot_matrix_single(
            self.ICNs[:, :, bestICN], dataType='ICA', fig=fig, ax=ax[1])
        ax[0].set_title('ICN coeffs for patient {} in {} band'.format(
            self.patientID, self.band), fontsize=12)
        ax[1].set_title('Best ICN: {}'.format(bestICN+1), fontsize=12)
        plt.show()

        return 'ICN with highest coeff: {}'.format(bestICN+1)

    def getCoeffs(self):

        if not hasattr(self, 'modelOut'):
            print('Need to run model first')

        return self.modelOut.coef_

    def getTestFeatures(self, calcVar=None, window=None):
        """
        Calculate features for each ICN over all time in order to test model on specific time points
        """

        if calcVar is None:
            calcVar = self.modelIn['calcVar']
        if window is None:
            window = self.modelIn['window']

        if not hasattr(self, 'ICNprojs'):
            self.loadICN('all')

        if calcVar == True:
            allFeatures, allFeatures_tAxis = IMSmodule.matrix_variance(
                self.ICNprojs, self.ICNtaxis, window)
        elif calcVar == False:
            allFeatures, allFeatures_tAxis = self.ICNproj, self.ICNtaxis

        return allFeatures, allFeatures_tAxis


def orthoregress(x, y):
    """
    Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c, nan, nan, nan]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    """
    linreg = linregress(x, y)
    mod = Model(f)
    dat = Data(x, y)
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()

    m = out.beta[0]
    c = out.beta[1]

    return m, c


def f(p, x):
    """
    Basic linear regression 'model' for use with ODR
    """
    return (p[0] * x) + p[1]
