# 2015-07-07, LKirkby

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd

from scipy.stats import mode


def calculate_coh_variance(cohMatrix, tAxis, window=60, center=True):
    """
    Calculate variance of coherence over sliding window, width 'window' in sec (default: 60sec)
    """

    tInt = mode(np.diff(tAxis))[0][0]
    intv = window/tInt
    offset = int(intv/2)

    cohvarMatrix = np.zeros_like(cohMatrix)

    if len(np.squeeze(cohMatrix).shape) > 1:
        nElecs_reg1 = cohMatrix.shape[0]
        nElecs_reg2 = cohMatrix.shape[1]
    else:
        nElecs_reg1 = 1
        nElecs_reg2 = 1

    if (nElecs_reg1 > 1) or (nElecs_reg2 > 1):
        # Define pairs over which to calculate coherence variances (list of tuples)
        ijPairs = []

        for i in range(nElecs_reg1):
            for j in range(nElecs_reg2):
                ijPairs.append((i, j))

        for pair in ijPairs:

            try:
                cohvar = pd.Series(cohMatrix[pair[0], pair[1]]).rolling(
                    window=int(intv), center=center).var()
            except:
                cohvar = pd.rolling_var(
                    cohMatrix[pair[0], pair[1]], int(intv), center=center)

            cohvarMatrix[pair[0], pair[1], :] = cohvar

            # if nElecs_reg1 == nElecs_reg2:
            #     cohvarMatrix[pair[1], pair[0], :] = cohvar

        cohvarMatrix = np.array(cohvarMatrix[:, :, offset:-offset])

    elif (nElecs_reg1 == 1) and (nElecs_reg2 == 1):
        try:
            cohvar = pd.Series(cohMatrix).rolling(
                window=int(intv), center=center).var()
        except:
            cohvar = pd.rolling_var(cohMatrix, int(intv), center=center)
        cohvarMatrix = np.array(cohvar[offset:-offset])

    cohvartAxis = tAxis[offset:-offset]

    return cohvarMatrix, cohvartAxis


def score_vs_feature(score, scoreTimes, feature, featuretAxis, deltaT=0, interval=600, displaytOut=False):
    """
    Plot feature of interest vs. score of interest (eg pain score, IMS), averaging feature 'deltaT' seconds before score time point over 'interval' time window.
    (generalized version of IMS_vs_feature function below...)
    """

    if displaytOut == True:
        if deltaT < 0:
            print('Averaging '+str(abs(deltaT)) +
                  's before score over '+str(interval)+'s time window.')
        elif deltaT >= 0:
            print('Averaging '+str(abs(deltaT)) +
                  's after score over '+str(interval)+'s time window.')

    # Determine times over which to average:
    tInt = []

    for tt in scoreTimes:
        tStart = tt+deltaT
        tEnd = tStart+interval

        tInt.append((tStart, tEnd))

    scoreList = []

    for nn, tPair in enumerate(tInt):

        tInds = np.where((featuretAxis >= tPair[0]) & (
            featuretAxis <= tPair[1]))[0]

        if len(tInds) > 0:
            meanVal = float(np.mean(feature[tInds]))
            stdVal = float(np.std(feature[tInds]))
            scoreList.append(
                {'score': score[nn], 'meanVal': meanVal, 'stdVal': stdVal})
        # else:
        #    print 'No data around IMS point '+str(nn)

    cols = ['score', 'meanVal', 'stdVal']
    scoreDataframe = pd.DataFrame(scoreList, columns=cols)

    return scoreDataframe


def IMS_vs_feature(IMS, IMStimes, feature, featuretAxis, deltaT=0, interval=600, displaytOut=False):
    """
    Plot feature of interest vs. IMS, averaging feature 'deltaT' seconds before IMS over 'interval' time window.
    """

    if displaytOut == True:
        if deltaT < 0:
            print('Averaging '+str(abs(deltaT)) +
                  's before IMS over '+str(interval)+'s time window.')
        elif deltaT >= 0:
            print('Averaging '+str(abs(deltaT)) +
                  's after IMS over '+str(interval)+'s time window.')

    # Determine times over which to average:
    tInt = []

    for tt in IMStimes:
        tStart = tt+deltaT
        tEnd = tStart+interval

        tInt.append((tStart, tEnd))

    IMSlist = []

    for nn, tPair in enumerate(tInt):

        tInds = np.where((featuretAxis >= tPair[0]) & (
            featuretAxis <= tPair[1]))[0]

        if len(tInds) > 0:
            meanVal = float(np.mean(feature[tInds]))
            stdVal = float(np.std(feature[tInds]))
            IMSlist.append(
                {'IMS': IMS[nn], 'meanVal': meanVal, 'stdVal': stdVal})
        # else:
        #    print 'No data around IMS point '+str(nn)

    cols = ['IMS', 'meanVal', 'stdVal']
    IMSdataframe = pd.DataFrame(IMSlist, columns=cols)

    return IMSdataframe


def matrix_variance(matIn, tAxis, window=60):
    """
    Calculate variance of matrix over sliding window, width 'window' in sec (default: 60sec)
    """

    tInt = mode(np.diff(tAxis))[0][0]
    intv = window/tInt

    matIn = np.atleast_2d(matIn)
    varMat = np.zeros_like(matIn)

    for rr, row in enumerate(matIn):
        varMat[rr] = pd.Series(row).rolling(
            window=int(intv), center=True).var()

    offset = int(intv/2)
    vartAxis = tAxis[offset:-offset]

    varMat = varMat[:, offset:-offset]
    varMat = np.nan_to_num(varMat)

    return varMat, vartAxis
