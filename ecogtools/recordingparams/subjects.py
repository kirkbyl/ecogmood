# 2015-07-08, LKirkby
# @kirkbyl


import matplotlib.pylab as plt
import numpy as np
import datetime
import time
import itertools
import collections
import pandas as pd
import cycler
import glob
import os

from ecogtools.recordingparams import elecs as electrodeinfo
from ecogtools.recordingparams import psych as moodinfo
from ecogtools.tools import loaddata, utilities as utils
from ecogtools.visualization import plotcoherence
from ecogtools.analysis import coherences as cohmodule
from ecogtools.preprocess import filters as fltrs


class Patient(object):
    """
    Information relevant to patient

    Inputs:
        - ID: str, patientID
    """

    def __init__(self, ID, dataPath=None):

        self.ID = ID
        self.dataPath = dataPath

        print('Loading '+str(self))
        self.getBrainRegions()
        self.getIMS()
        self.getPaths()

    def __str__(self):

        return 'Patient '+self.ID

    def getID(self):

        return self.ID

    def getBrainRegions(self):

        try:
            self.brainRegions = electrodeinfo.electrode_locs_dataframe(self.ID)
            self.brainRegionsDict = electrodeinfo.electrode_locs(self.ID)[
                'brainRegions']
            self.brainRegionsDict_abbr = electrodeinfo.electrode_locs(self.ID)[
                'brainRegions_abbr']
        except:
            print '\tNo electrode info for '+self.ID

        return self.brainRegions

    def getIMS(self):

        try:
            self.IMS = moodinfo.IMS_timepoints(self.ID)
        except:
            self.IMS = None
            print '\tNo IMS for '+self.ID

        return self.IMS

    def getPaths(self):

        if self.dataPath is not None:
            self.signalsPath = self.dataPath+'/Signals/'+self.ID
            self.cohPath = self.dataPath+'/Coherence/'+self.ID
            self.ICNpath = self.dataPath+'/ICN/'+self.ID
        else:
            print('No link to dataPath')

    def getPsychScores(self):

        try:
            self.psychScores = moodinfo.psych_scores(self.ID)
            print self.psychScores
        except:
            print '\tNo psych scores for '+self.ID

        return self.psychScores

    def getAnalysisFiles(self):

        try:
            self.signalInfo = loaddata.signals_dataframe(
                self.ID, self.signalsPath)
        except:
            print("\tUnable to load signals info for "+self.ID)
        try:
            self.cohInfo = loaddata.coherence_dataframe(self.ID, self.cohPath)
        except:
            print("\tUnable to load coherence info for "+self.ID)
        try:
            self.icnInfo = loaddata.icn_dataframe(self.ID, self.ICNpath)
        except:
            print("\tUnable to load ICN info for "+self.ID)

    def loadSignals(self, tStamp=None, deltaIdx=0):
        """
        timeStamp can be pandas Timestamp object, datetime object, 6-tuple datetimes or UTC time integer
        eg. 2:30:47 pm on 17 January 2015 in 6-tuple is: (2015, 1, 17, 14, 30, 47)
        """

        self.getAnalysisFiles()

        sigout, fileID = loaddata.reload_signals_from_timestamp(
            self.signalInfo, timeStamp=tStamp, deltaIdx=deltaIdx)

        return sigout, fileID

    def loadCoherence(self, band, tBin, matType='average'):
        """
        Load coherence matrices:
            - band: frequency band
            - tBin: time bin over which coherence is calculated
            - matType = 'full' (all electrodes) or 'average' (averaged across brain regions)
        """

        self.getAnalysisFiles()
        cohOut, fileID = loaddata.reload_coherence(
            self.cohInfo, band, tBin, matType=matType)

        return cohOut

    def loadICN(self, band):
        """
        Load ICN info:
            - band: frequency band
        """

        self.getAnalysisFiles()
        icnOut, fileID = loaddata.reload_icn(self.icnInfo, band)

        return icnOut

    def check64Grid(self):

        try:
            cohOut = self.loadCoherence('beta', 10, matType='full')
            dataMatrix = cohOut['coherence']
        except:
            print('Could not load coherence data for {0}'.format(self.ID))
            dataMatrix = None

        self.no64Grid = cohmodule.is_grid_removed(self.ID, dataMatrix)

        return self.no64Grid

    def chooseElectrodes(self, elecAbbrs, no64Grid=False, excludeGrids=False):

        if not hasattr(self, 'no64Grid'):
            self.no64Grid = self.check64Grid()

        regs, regsNew = electrodeinfo.choose_electrodes(
            self.getID(), elecAbbrs, no64Grid=self.no64Grid, excludeGrids=excludeGrids)

        return regs, regsNew

    def getRegionInds(self, elecAbbrs):

        regInds = electrodeinfo.get_region_index(self.getID(), elecAbbrs)

        return regInds

    def subsetMatrix(self, elecAbbrs, band, tBin, dataType='coherence'):
        """
        dataType can be 'coherence' or 'phase'
        """

        regs, regsNew = self.chooseElectrodes(elecAbbrs)
        subsetInds = [
            kk for kk in itertools.chain.from_iterable(regs.values())]
        if dataType == 'coherence':
            matOut = self.loadCoherence(band, tBin, matType='full')
        elif dataType == 'phase':
            matOut = self.loadPhase(band, tBin, matType='full')
        subsetMat = cohmodule.subset_matrix(matOut[dataType], subsetInds)

        return subsetMat, regsNew


def create_patients(patientIDlist, dataPath=None):
    """
    Creates dictionary of Patient objects define by {'patientID': Patient('patientID')}

    Inputs:
        - patientIDlist: list of string of patientIDs eg ['EC71', 'EC77', 'EC79']
    """

    #print('Loading subjects\n')

    patientDict = collections.OrderedDict()

    for pID in patientIDlist:
        patient = Patient(pID, dataPath=dataPath)
        patientDict[pID] = patient

        exec(pID + "= patient")

    # print('\nDone')

    return patientDict


class EcogSignals(object):
    """
    Analysis class for easy plotting of ecog signals
    """

    def __init__(self, pID, dataPath=None):
        """
        Inputs:
            - pID: string of patient ID or instance of Patient class
        """

        if isinstance(pID, str):
            self.patientID = pID
            self.patient = Patient(pID, dataPath=dataPath)
        else:
            self.patientID = pID.ID
            self.patient = pID

        self.allSignals = loaddata.signals_dataframe(
            self.patient.getID(), self.patient.signalsPath)
        self.brainRegions = self.patient.getBrainRegions()

        return

    def loadSignals(self, tStamp=None, deltaIdx=0):

        sigout, fileID = self.patient.loadSignals(
            tStamp=tStamp, deltaIdx=deltaIdx)

        self.signals = sigout['signals']
        self.sampFq = sigout['sampFq']
        self.tAxis = sigout['tAxis']

        return sigout, fileID

    def getSubsetSignals(self, brainReg=None, excludeGrids=True, tStamp=None, nMin=5, deltaIdx=0, center=False):
        """
        - brainReg: string representing brain region of interest
        - nMin: number of minutes of data to plot
        - tStamp: time around which to plot. tStamp can be pandas Timestamp object, datetime object, 6-tuple datetimes or UTC time integer
            eg. 2:30:47 pm on 17 January 2015 in 6-tuple is: (2015, 1, 17, 14, 30, 47)
        """

        sigout, fileID = self.loadSignals(tStamp=tStamp, deltaIdx=deltaIdx)
        regs = self.getElecs(brainReg, excludeGrids=excludeGrids)

        elecs = list(itertools.chain.from_iterable(regs.values()))

        if self.signals is not None:
            self.subsetSignals, self.subsetTime = loaddata.get_subset_signals(
                self.signals, self.tAxis['UTC'], elecs, tStamp=tStamp, nMin=nMin, center=center)
        else:
            self.subsetSignals = None
            self.subsetTime = None

        return self.subsetSignals, self.subsetTime

    def getElecs(self, brainReg=None, excludeGrids=True):
        """
        - brainReg: list or string (comma separated) of abbreviations of brain regions of interest
          eg. brainReg = ['AM', 'HPC', 'OFC']  or 'AM, HPC, OFC', to choose amygdala, hippocampus and OFC electrodes  
          can also be:
              - None (to enter ROI at command line) or
              - 'all' (to choose all electrodes)
        """

        if brainReg == None:
            brainReg = raw_input("Enter region of interest. Choose from " +
                                 str(self.patient.brainRegionsDict.keys())+" or 'all': ")

        if isinstance(brainReg, str):
            brainReg = utils.str_to_list(brainReg)

        regs = collections.OrderedDict()

        if len(brainReg) == 1:
            breg = brainReg[0]

            if breg == 'all':
                regs = self.patient.brainRegionsDict
            elif breg in self.patient.brainRegionsDict:
                elecs = self.patient.brainRegionsDict[breg]
                regs[breg] = elecs
            elif breg in self.patient.brainRegionsDict_abbr:
                elecs = self.patient.brainRegionsDict_abbr[breg]
                regs[breg] = elecs
        else:
            regs, regsNew = electrodeinfo.choose_electrodes(
                self.patient.ID, brainReg, excludeGrids=excludeGrids)

        return regs

    def filterSignals(self, filtFqs):
        """
        - filtFqs: 2-element list [lowcut, highcut]
        """

        if not hasattr(self, 'subsetSignals'):
            print('Need to create subsetSignals before filtering')
            self.signalsFilt = None
        else:
            self.signalsFilt = fltrs.butter_bandpass_filter(
                self.subsetSignals, self.sampFq, filtFqs[0], filtFqs[1])

        return self.signalsFilt

    def totalDuration(self):

        return self.allSignals.iloc[-1]['endT'] - self.allSignals.iloc[0]['startT']

    def sampleTimeAxis(self, nSamples=100):
        """
        Chooses nSamples equally-spaced times across recording duration
        """

        tTot = self.totalDuration()
        deltaT = tTot/nSamples
        print('Sampling time axis every '+str(deltaT)[:-7]+' hours')

        # Start 1 hour into recording
        tStart = self.allSignals.iloc[0]['startT'] + \
            datetime.timedelta(minutes=60)

        timeSamples = []
        for ii in range(nSamples):
            timeSamples.append(tStart+(ii*deltaT))

        if timeSamples[-1] > self.allSignals.iloc[-1]['endT']:
            del timeSamples[-1]

        return timeSamples


class SignalArray(object):

    def __init__(self, sigs, tAxis):

        self.signals = sigs
        self.tAxis = tAxis
        self.sampFq = round(len(tAxis)/(tAxis[-1]-tAxis[0]))

    def plotSignals(self, indices='all'):

        if indices == 'all':
            indices = range(0, len(self.signals))

        q = plotsignals.plot_data_scrollwheel(
            self.signals, self.tAxis, indices)

        return q

    def calculatePower(self, zscore=True, plotPower=False):

        self.f, self.Pxx = pwrmodule.powerspectrum(
            self.signals, self.sampFq, zscore=zscore)

        if plotPower == True:
            self.plotPower(norm=zscore)

    def plotPower(self, norm=True, ylim=[]):

        self.calculatePower(zscore=norm, plotPower=False)

        colors = plt.cm.spectral(np.linspace(0, 0.9, len(self.Pxx)))
        plt.gca().set_prop_cycle(cycler.cycler('color', colors))

        fig, ax = plt.subplots()
        for ii, P in enumerate(self.Pxx):
            plt.semilogy(self.f, P, label=str(ii), color=colors[ii])
            if ylim != []:
                plt.ylim(ylim)
        ax.set_xlabel('Frequency, [Hz]')
        ax.set_ylabel('Spectral density')
        plt.legend(fontsize='small')
