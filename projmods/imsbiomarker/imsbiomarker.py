# @kirkbyl, 21 July 2016
"""
Class for biomarker analysis: neural feature that correlates with IMS
Pooling across patients
"""

import os
import numpy as np
import sys
import argparse
import time
import matplotlib.pylab as plt
from scipy.stats import linregress
import itertools
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

from ecogtools.analysis import IMScorr as IMSmodule
from ecogtools.visualization import plotcorrelations
from ecogtools.recordingparams import subjects as subj, elecs as electrodeinfo, psych as psychmodule
from ecogtools.tools import utilities as utils


class PooledPatients(object):
    """
    Collect data (power or coherence) across list of patients given by patientIDs for pooling data
    """

    def __init__(self, patientIDs, dataPath=None):

        self.patients = subj.create_patients(patientIDs, dataPath=dataPath)
        self.patientIDs = patientIDs

        return

    def addPatient(self, newID):

        if newID not in self.patientIDs:
            self.patientIDs.append(newID)
            self.patients = subj.create_patients(self.patientIDs)

        return

    def removePatient(self, subID):

        if subID in self.patientIDs:
            self.patientIDs.remove(subID)
            self.patients = subj.create_patients(self.patientIDs)

        return

    def loadAll(self, band, regs, dtype='coherence', tBin=10, matType='average'):
        """
        ## NOTE: only works for matType = 'average', ie matType='full' not implemented
        """

        if not hasattr(self, 'data'):
            self.data = {}

        datatype = dtype+'-'+band

        if datatype not in self.data:
            self.data[datatype] = {}

        for pID in self.patientIDs:

            if pID not in self.data[datatype]:
                self.data[datatype][pID] = {}

            if regs not in self.data[datatype][pID]:
                print('Loading {0} {1} data for {2}'.format(
                    regs, datatype, pID))
                self.data[datatype][pID][regs], self.data[datatype][pID]['tAxis'] = load_data(
                    self.patients[pID], dtype, band, regs, tBin=10, matType='average')

            else:
                print('Already have {0} {1} data for {2}'.format(
                    regs, datatype, pID))

        return

    def calcVarAll(self, band, regs, dtype='coherence', window=60):

        self.loadAll(band, regs, dtype=dtype)

        if not hasattr(self, 'dataVar'):
            self.dataVar = {}

        datatype = dtype+'-'+band
        if datatype not in self.dataVar:
            self.dataVar[datatype] = {}

        for pID in self.patientIDs:

            self.dataVar[datatype][pID] = {}

            print('Calculating variance for {0} {1} data for {2}'.format(
                regs, datatype, pID))
            if dtype == 'coherence':
                self.dataVar[datatype][pID][regs], self.dataVar[datatype][pID]['tAxis'] = IMSmodule.calculate_coh_variance(
                    self.data[datatype][pID][regs], self.data[datatype][pID]['tAxis'], window=window)
            elif dtype == 'power':
                self.dataVar[datatype][pID][regs], self.dataVar[datatype][pID]['tAxis'] = IMSmodule.calculate_power_variance(
                    self.data[datatype][pID][regs], self.data[datatype][pID]['tAxis'], window=window)

        return

    def getIMSfeatures(self, band, regs, dtype='coherence', tBin=10, matType='average', calcVar=True, deltaT=-600, interval=1200, window=60, plotCorr=True):
        """
        Collecting coherence or power features to regress against IMS, for pooling across patients
        """

        self.IMS_all = []
        self.feature_all = []

        if isinstance(regs, str):
            regList = [rr.strip() for rr in regs.split(',')]
        elif isinstance(regs, list):
            regList = regs

        # Load all data:
        datatype = dtype+'-'+band
        if calcVar == False:
            self.loadAll(band, regs, dtype=dtype, tBin=tBin, matType=matType)
        elif calcVar == True:
            self.calcVarAll(band, regs, dtype=dtype, window=window)

        for pID in self.patientIDs:

            if calcVar == True:
                feature = self.dataVar[datatype][pID][regs]
                featuretAxis = self.dataVar[datatype][pID]['tAxis']
            elif calcVar == False:
                feature = self.data[datatype][pID][regs]
                featuretAxis = self.data[datatype][pID]['tAxis']

            if feature is not None:
                IMSpts = list(self.patients[pID].IMS['IMS'])
                IMStimes = [time.mktime(tt.timetuple())
                            for tt in self.patients[pID].IMS['IMStimes']]

                IMSdataframe = IMSmodule.IMS_vs_feature(
                    IMSpts, IMStimes, feature, featuretAxis, deltaT=deltaT, interval=interval)

                x = IMSdataframe['meanVal']
                y = IMSdataframe['IMS']

                # Set labels for figure:

                self.figLabels = {}

                if calcVar == True:
                    varTitle = 'variance'
                elif calcVar == False:
                    varTitle = ''

                xlabel = '{0} {1}'.format(dtype.capitalize(), varTitle)
                ylabel = 'IMS'
                title = 'Patient: {0}, Band: {1}, Regions: {2}'.format(
                    pID, band, '-'.join(regList))

                self.figLabels['title'] = 'Regions: {0}, Band: {1}'.format(
                    '-'.join(regList), band)
                self.figLabels['xlabel'] = xlabel
                self.figLabels['ylabel'] = ylabel

                slope, intercept, r, p, stderr = linregress(x, y)

                if plotCorr == True:

                    fig, ax = plt.subplots()
                    plotcorrelations.scatter_regplot(x, y, ax=ax)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    plt.show()

                # Combine across patients for pooling:

                self.IMS_all.append(list(y))
                self.feature_all.append(list(x))

        return self.feature_all, self.IMS_all

    def plotPooled(self, band, regs, normType='z-score', showPlot=True, linearFit=True, plotRange=False, pctileRange=[25, 75], dtype='coherence', tBin=10, matType='average', calcVar=True, deltaT=-600, interval=1200, window=60, plotCorr=False, legendLoc=0):
        """
        Pooled plot of IMS vs feature

        - normType: normalization, can be 'z-score' (both IMS and feature are z-scored within patient) or 'normIMS' (IMS normalized between 0-100, no normalization of feature) or 'None'
        """
        self.getIMSfeatures(band, regs, dtype=dtype, tBin=tBin, matType=matType, calcVar=calcVar,
                            deltaT=deltaT, interval=interval, window=window, plotCorr=plotCorr)

        datatype = dtype+'-'+band
        if calcVar == True:
            featureDict = self.dataVar[datatype]
        elif calcVar == False:
            featureDict = self.data[datatype]
        featureRange = self.getFeatureRange(
            featureDict, regs, percentile=pctileRange)

        IMS_norm = []
        feature_norm = []

        IMS_join = []
        feature_join = []

        for pID, ims, feat in zip(self.patientIDs, self.IMS_all, self.feature_all):

            if normType == 'z-score':

                i_norm = [((II-np.mean(ims))/np.std(ims)) for II in ims]
                f_norm = [((FF-np.mean(feat))/np.std(feat)) for FF in feat]

                IMS_norm.append(i_norm)
                feature_norm.append(f_norm)

                addxLabel = '[z-score]'
                addyLabel = '[z-score]'

            elif normType == 'normIMS':

                if pID == 'EC79' or pID == 'EC80' or pID == 'EC81':
                    IMSrange = [-69, 69]
                elif pID == 'EC84' or pID == 'EC88' or pID == 'EC90':
                    IMSrange = [-15, 15]
                else:
                    IMSrange = [-72, 72]

                i_norm = [
                    int(round(((float(II)-IMSrange[0])/(IMSrange[1]-IMSrange[0]))*100)) for II in ims]
                f_norm = feat

                IMS_norm.append(i_norm)
                feature_norm.append(f_norm)

                addxLabel = ''
                addyLabel = '[normalized]'

            elif normType == 'None':

                i_norm = ims
                f_norm = feat

                IMS_norm.append(i_norm)
                feature_norm.append(f_norm)

                addxLabel = ''
                addyLabel = ''

        IMS_join = list(itertools.chain.from_iterable(IMS_norm))
        feature_join = list(itertools.chain.from_iterable(feature_norm))

        if showPlot == True:
            # Plot different markers/colors
            nDatasets = len(self.IMS_all)

            markers = itertools.cycle(('o', '+', '*', '2'))
            sizes = itertools.cycle((40, 140, 100, 140))
            colors = cm.brg(np.linspace(0, 0.6, nDatasets))
            #colors = cm.rainbow(np.linspace(0, 1, nDatasets))

            fig, ax = plt.subplots(figsize=(8, 8))
            for vv, ii, cc, mm, ss, pp, ff in zip(feature_norm, IMS_norm, colors, markers, sizes, self.patientIDs, featureRange):
                plt.scatter(vv, ii, marker=mm, s=ss, c=rgb2hex(
                    cc), edgecolor=rgb2hex(cc), label=pp)
                if normType == 'normIMS' and plotRange == True:
                    plt.errorbar(np.median(vv), np.mean(ii), xerr=[[np.median(
                        vv)-ff[0]], [ff[-1]]], yerr=[[np.std(ii)], [np.std(ii)]], fmt='k--o', ecolor='k')
            if linearFit == True:
                plotcorrelations.scatter_regplot(np.array(feature_join), np.array(
                    IMS_join), ax=ax, scatter_kws={'s': 0}, color='k')
            ax.set_xlabel(self.figLabels['xlabel']+' '+addxLabel)
            ax.set_ylabel(self.figLabels['ylabel']+' '+addyLabel)
            ax.set_title(self.figLabels['title'])
            plt.legend(fontsize='small', loc=legendLoc)
            plt.show()
        else:
            fig = None
            ax = None

        slope, intercept, r, p, stderr = linregress(feature_join, IMS_join)

        return {'x': feature_join, 'y': IMS_join, 'p': p, 'r2': r**2, 'fig': fig, 'ax': ax}

    def plotHeatmap(self, band, regs, deltaTs=[-1800, -600, -300, 0, 300, 600], intervals=[10, 60, 300, 600, 1200, 1800], dtype='coherence', tBin=10, matType='average', calcVar=True, window=60, showCorrPlot=False, clim=[-12, 0]):
        """
        2-D heat map of p-values for fit of pooled data for range of different deltaT and interval
        """

        self.pMatrix = np.zeros((len(deltaTs), len(intervals)))
        self.rMatrix = np.zeros((len(deltaTs), len(intervals)))
        for dd, deltaT in enumerate(deltaTs):
            print('deltaT: {0}'.format(deltaT))

            for ii, interval in enumerate(intervals):
                #print('deltaT: {0}, interval: {1}'.format(deltaT, interval))

                plotOut = self.plotPooled(band, regs, deltaT=deltaT, interval=interval, plotCorr=False, dtype=dtype,
                                          tBin=tBin, matType=matType, calcVar=calcVar, window=window, showPlot=showCorrPlot)

                self.pMatrix[dd, ii] = np.log10(plotOut['p'])
                self.rMatrix[dd, ii] = plotOut['r2']

        fig, ax = plt.subplots(figsize=(8, 7))
        plt.imshow(self.pMatrix, cmap='bone',
                   interpolation='None', aspect='auto', clim=clim)
        plt.colorbar()
        plt.yticks(np.arange(len(deltaTs)), deltaTs)
        plt.xticks(np.arange(len(intervals)), intervals)
        plt.title('log(p-value) for best fit')
        plt.ylabel('Delta t from IMS point [s]')
        plt.xlabel('Interval for average [s]')
        plt.show()

        return

    def getFeatureRange(self, dataDict, regs, percentile=[25, 50, 75]):
        """
        Collect range of coherence or power features for each patient
        """

        featureRange = []

        for pID in self.patientIDs:
            ff = dataDict[pID][regs]

            pctile = [np.percentile(ff, pc) for pc in percentile]
            featureRange.append(pctile)

        return featureRange

    def plotTraits(self, band, regs, pctileRange=[25, 50, 75], dtype='coherence', tBin=10, matType='average', calcVar=True, window=60):
        """
        Plot feature vs. trait (BDI, BAI, PHQ9, GAD7, Rum)
        """

        # Load all data:
        datatype = dtype+'-'+band
        if calcVar == False:
            self.loadAll(band, regs, dtype=dtype, tBin=tBin, matType=matType)
            featureDict = self.data[datatype]
        elif calcVar == True:
            self.calcVarAll(band, regs, dtype=dtype, window=window)
            featureDict = self.dataVar[datatype]

        featureRange = np.array(self.getFeatureRange(
            featureDict, regs, percentile=pctileRange))

        for trait in ['BDI', 'BAI', 'PHQ9', 'GAD7', 'Rum']:
            traitAll = []

            for pID in self.patientIDs:
                psychScores = psychmodule.psych_scores(pID)
                traitAll.append(psychScores[trait])

            markers = itertools.cycle(('.', '+', '*', '2'))
            colors = cm.brg(np.linspace(0, 0.6, len(traitAll)))
            fig, ax = plt.subplots()
            plt.errorbar(traitAll, featureRange[:, 1], yerr=[
                         featureRange[:, 1]-featureRange[:, 0], featureRange[:, -1]], fmt='ko', ecolor='k')
            plt.title(trait)

        return


def load_data(patient, dtype, band, regs, tBin=10, matType='average'):

    inds = electrodeinfo.get_region_index(patient.ID, regs)
    reglist = utils.str_to_list(regs)

    try:
        if dtype == 'coherence':
            cohOut = patient.loadCoherence(band, tBin, matType=matType)
            data_all, datatAxis = cohOut['coherence'], cohOut['tAxis']['UTC']

            if reglist[0] == reglist[1]:
                data = data_all[inds.values()[0], inds.values()[0], :]

            else:
                # NOTE assumes only 1 region with electrodes in each abbr specified in regs...
                data = data_all[inds.values()[0], inds.values()[1], :]

        elif dtype == 'power':
            powOut = patient.loadPower(band, tBin, matType=matType)
            data_all, datatAxis = powOut['power'], powOut['tAxis']['UTC']

            # NOTE assumes only 1 region with electrodes in each abbr specified in regs...
            data = data_all[inds.values()[0]]

    except:
        print('Could not load {0} data for {1}'.format(dtype, patient.ID))
        data = None
        datatAxis = None

    return data, datatAxis


def find_feature(patient, dtype, band, regs, tBin=10, matType='average', calcVar=True, window=60):
    """
    Get power or coherence time series corresponding to patient, band and regions of interest
    """

    reglist = utils.str_to_list(regs)
    inds = electrodeinfo.get_region_index(patient.ID, regs)

    try:
        if dtype == 'coherence':
            cohOut = patient.loadCoherence(band, tBin, matType=matType)

            if calcVar == True:
                feature_all, featuretAxis = IMSmodule.calculate_coh_variance(
                    cohOut['coherence'], cohOut['tAxis']['UTC'], window=window)
            else:
                feature_all, featuretAxis = cohOut['coherence'], cohOut['tAxis']['UTC']

            if reglist[0] == reglist[1]:
                feature = feature_all[inds.values()[0], inds.values()[0], :]

            else:
                # NOTE assumes only 1 region with electrodes in each abbr specified in regs...
                feature = feature_all[inds.values()[0], inds.values()[1], :]

        elif dtype == 'power':
            powOut = patient.loadPower(band, tBin, matType=matType)

            if calcVar == True:
                feature_all, featuretAxis = IMSmodule.calculate_power_variance(
                    powOut['power'], powOut['tAxis']['UTC'], window=window)
            else:
                feature_all, featuretAxis = powOut['power'], powOut['tAxis']['UTC']

            # NOTE assumes only 1 region with electrodes in each abbr specified in regs...
            feature = feature_all[inds.values()[0]]
    except:
        print('Could not load {0} data for {1}'.format(dtype, patient.ID))
        feature = None
        featuretAxis = None

    return feature, featuretAxis
