# 2015-07-06, LKirkby

import numpy as np
import os
import pandas as pd
import datetime
import time
import glob
import matplotlib.dates as mdates
try:
    import cPickle as pickle
except:
    import pickle

from ecogtools.tools import utilities as utils


def coherence_dataframe(patientID, cohDir):
    """
    Creates dataframe for easy loading of coherence matrices
    """

    # Check if cohdir has patient ID name in it:
    lastDir = cohDir.split('/')[-1]

    if lastDir == patientID:
        npyDir = cohDir
    else:
        npyDir = cohDir+'/'+patientID

    npyFiles = sorted(glob.glob(npyDir+'/*.npy'))

    # List to convert to pandas dataframe:
    dfRow = []

    for npyFile in npyFiles:

        npyfilename = os.path.split(npyFile)[-1]
        filename = npyfilename.split('.')[0]

        pID = filename.split('_')[0]
        band = filename.split('_')[-1]
        matrixtype = filename.split('_')[1]
        timebin = filename.split('_')[2]

        #coherences = np.load(npyFile, mmap_mode = 'r')

        dfRow.append({'patientID': pID, 'fqBand': band,
                      'tBin': timebin, 'type': matrixtype, 'filePath': npyFile})

    cols = ['patientID', 'fqBand', 'tBin', 'type', 'filePath']
    cohInfo = pd.DataFrame(dfRow, columns=cols)

    return cohInfo


def signals_dataframe(patientID, signalsDir):
    """
    Creates dataframe for easy loading of signals
    """

    # Check if signalsDir has patient ID name in it:
    lastDir = signalsDir.split('/')[-1]

    if lastDir == patientID:
        npyDir = signalsDir
    else:
        npyDir = signalsDir+'/'+patientID

    signalFiles = sorted(glob.glob(npyDir+'/*signal.npy'))

    # List to convert to pandas dataframe:
    dfRow = []

    for ss, signalFile in enumerate(signalFiles):

        filename = os.path.split(signalFile)[-1].split('_signal')[0]

        timeFile = signalFile.replace('_signal.npy', '_time.npy')

        tAxis = np.load(timeFile, mmap_mode='r')
        startT = datetime.datetime.fromtimestamp(tAxis[0])
        endT = datetime.datetime.fromtimestamp(tAxis[-1])
        sampFq = float(round(1./(np.mean(np.diff(tAxis[0:10])))))

        dfRow.append({'filename': filename, 'startT': startT,
                      'endT': endT, 'sampFq': sampFq, 'signalsFile': signalFile})

    cols = ['filename', 'startT', 'endT', 'sampFq', 'signalsFile']
    signalInfo = pd.DataFrame(dfRow, columns=cols)
    try:
        signalInfo.sort_values('startT', inplace=True)
    except:
        signalInfo.sort('startT', inplace=True)

    signalInfo.index = range(0, len(signalInfo))

    return signalInfo


def reload_signals_from_timestamp(patientSigDF, timeStamp=None, deltaIdx=0):
    """
    Reload signals from processed npy file corresponding to timepoint of interest (or file that is deltaIdx files ahead or beind timepoint of interest)
    timeStamp can be pandas Timestamp object, datetime object, 6-tuple datetimes or UTC time integer
    eg. 2:30:47 pm on 17 January 2015 in 6-tuple is: (2015, 1, 17, 14, 30, 47)

    patientSigDF is the patient specific dataframe that is generated undert the subjects.Patient class in ecogtools.recordingparams
    """

    if timeStamp is None:
        loadIdx = 0
    else:
        if isinstance(timeStamp, pd.tslib.Timestamp) or isinstance(timeStamp, datetime.datetime):
            timepoint = timeStamp
        elif isinstance(timeStamp, int) or isinstance(timeStamp, float) or isinstance(timeStamp, tuple):
            timepoint = utils.convert_timestamp([timeStamp])['datetime']

        sigIdx_temp = np.where((patientSigDF['startT'] <= timepoint) & (
            patientSigDF['endT'] >= timepoint))[0]
        if len(sigIdx_temp) == 0:
            print('No data at time {0}'.format(timepoint))
            loadIdx = None
        else:
            sigIdx = np.where((patientSigDF['startT'] <= timepoint) & (
                patientSigDF['endT'] >= timepoint))[0][0]
            loadIdx = sigIdx+deltaIdx

    tAxis = {}

    if loadIdx is not None and loadIdx <= len(patientSigDF['signalsFile']):
        signals = np.load(patientSigDF['signalsFile'][loadIdx], mmap_mode='r')
        tAxis['UTC'] = np.load(patientSigDF['signalsFile'][loadIdx].replace(
            '_signal.npy', '_time.npy'), mmap_mode='r')

        # Seconds since start of recording
        nTimePoints = len(tAxis['UTC'])

        #sampFq = round((nTimePoints)/(tAxis['UTC'][-1]-tAxis['UTC'][0]))
        sampFq = patientSigDF['sampFq'][loadIdx]
        nSec = nTimePoints/float(sampFq)
        tAxis['sec'] = np.linspace(0, nSec, nTimePoints)

        # matplotlib time axis (days since 0001-01-01 UTC plus 1)

        startObj = datetime.datetime.fromtimestamp(tAxis['UTC'][0])
        endObj = datetime.datetime.fromtimestamp(tAxis['UTC'][-1])

        mplStart = mdates.date2num(startObj)
        mplEnd = mdates.date2num(endObj)
        tAxis['matplotlib'] = np.linspace(mplStart, mplEnd, nTimePoints)

        fileID = patientSigDF['signalsFile'][loadIdx]
    else:
        print('Index out of bounds')
        signals = None
        tAxis = None
        sampFq = None
        fileID = None

    return {'signals': signals, 'tAxis': tAxis, 'sampFq': sampFq}, fileID


def reload_coherence(patientCohDF, band, tBin, matType='average'):
    """
    Reload coherence matrices corresponding to specific fq band, time bin, or matType ('full' or 'average')
    patientCohDF is the patient specific dataframe that is generated undert the subjects.Patient class in ecogtools.recordingparams
    """

    cohOut, fileID = reload_matrix_from_dataframe(
        patientCohDF, band, tBin, dataName='coherence', matType=matType)

    return {'coherence': cohOut['data'], 'tAxis': cohOut['tAxis']}, fileID


def reload_power(patientPowerDF, band, tBin, matType='average'):
    """
    Reload power matrices corresponding to specific fq band, time bin, or matType ('full' or 'average')
    patientPowerDF is the patient specific dataframe that is generated undert the subjects.Patient class in ecogtools.recordingparams
    """

    powerOut, fileID = reload_matrix_from_dataframe(
        patientPowerDF, band, tBin, dataName='power', matType=matType)

    return {'power': powerOut['data'], 'tAxis': powerOut['tAxis']}, fileID


def reload_matrix_from_dataframe(patientDF, band, tBin, dataName='coherence', matType='average'):

    if dataName == 'coherence':
        dataAbbr = 'coh'
    elif dataName == 'phase':
        dataAbbr = 'phase'
    elif dataName == 'power':
        dataAbbr = 'power'

    if matType == 'average':
        ctype = dataAbbr+'Ave'
    elif matType == 'full':
        ctype = dataAbbr

    idx = np.where((patientDF['fqBand'] == band) & (
        patientDF['tBin'] == str(tBin)+'s') & (patientDF['type'] == ctype))[0][0]
    filename = patientDF['filePath'][idx]
    timeFile = filename.replace(band, 'time')

    tAxis = {}

    if idx <= len(patientDF['filePath']):
        data = np.load(filename, mmap_mode='r')
        tAxis['UTC'] = np.load(timeFile, mmap_mode='r')

        tConv = utils.convert_timestamp(tAxis['UTC'])
        tAxis['matplotlib'] = tConv['matplotlib']
        tAxis['datetime'] = tConv['datetime']

    else:
        print('Index out of bounds')

    fileID = patientDF['filePath'][idx]

    return {'data': data, 'tAxis': tAxis}, fileID


class DictToDot(object):
    """
    Convert a dictionary of paths into an object accessed via dot.notation
    """
    #__getattr__= dict.__getitem__

    def __init__(self, d):
        # self.__dict__.update(**dict((k, self.__parse(v))
        #                   for k, v in d.iteritems()))
        for k, v in d.iteritems():
            setattr(self, k, self.__parse(v))

    @classmethod
    def __parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.__parse(i) for i in v]
        else:
            return v


def icn_dataframe(patientID, icnDir):
    """
    Creates dataframe for easy loading of intrinsic connectivity networks
    """

    # Check if icnDir has patient ID name in it:
    lastDir = icnDir.split('/')[-1]

    if lastDir == patientID:
        dataDir = icnDir
    else:
        dataDir = icnDir+'/'+patientID

    folderIDs = ['coh', 'cohAve', 'ICN', 'proj', 'regs']

    # List to convert to pandas dataframe:
    dfRow = []

    for folderID in folderIDs:
        allfiles = sorted(
            glob.glob(dataDir+'/'+folderID+'/*'+folderID+'*.npy'))

        for ii, icnfile in enumerate(allfiles):

            band = icnfile.split('_')[-1].split('.')[0]

            dfRow.append(
                {'dataType': folderID, 'fqBand': band, 'filePath': icnfile})

    cols = ['dataType', 'fqBand', 'filePath']
    icnInfo = pd.DataFrame(dfRow, columns=cols)

    return icnInfo


def reload_icn(patientIcnDF, band):
    """
    Reload icn info corresponding to specific fq band

    patientIcnDF is the patient specific dataframe that is generated under the subjects.Patient class in ecogtools.recordingparams
    """

    folderIDs = ['coh', 'cohAve', 'ICN', 'proj', 'regs']
    icnOut = {}

    for folderID in folderIDs:
        icnIdx = np.where((patientIcnDF['fqBand'] == band) & (
            patientIcnDF['dataType'] == folderID))[0][0]
        loadFile = patientIcnDF['filePath'][icnIdx]

        if folderID == 'regs':
            icnOut[folderID] = np.load(loadFile)[()]
        else:
            icnOut[folderID] = np.load(loadFile, mmap_mode='r')

    timeIdx = np.where((patientIcnDF['fqBand'] == 'time') & (
        patientIcnDF['dataType'] == 'proj'))[0][0]
    timeFile = patientIcnDF['filePath'][timeIdx]

    icnOut['tAxis'] = np.load(timeFile)[()]

    return icnOut


def get_subset_signals(signalArray, tAxis, elecs, tStamp=None, nMin=5, center=False):
    """
    get subset signals from signalArray:
        - tAxis: corresponding time axis in UTC
        - elecs: electrodes of interest
        - tStamp: timestamp of interest
        - nMin: number of minutes of subset signals needed
        - center: centered around tStamp if True (-nMin/2:tStamp:nMin/2) and after if False (tStamp:nMin)
    """

    sampFq = round(len(tAxis)/(tAxis[-1]-tAxis[0]))
    startDT = utils.convert_timestamp([tAxis[0]])['datetime']
    endDT = utils.convert_timestamp([tAxis[-1]])['datetime']

    if tStamp is not None:
        # Convert tStamp to datetime object
        if isinstance(tStamp, pd.tslib.Timestamp) or isinstance(tStamp, datetime.datetime):
            timepoint = tStamp
        elif isinstance(tStamp, int) or isinstance(tStamp, float) or isinstance(tStamp, tuple):
            timepoint = utils.convert_timestamp([tStamp])['datetime']

        if center == True:
            startT = timepoint-datetime.timedelta(minutes=nMin/2.)
            endT = timepoint+datetime.timedelta(minutes=nMin/2.)
        else:
            startT = timepoint
            endT = timepoint+datetime.timedelta(minutes=nMin)
    else:
        startT = startDT
        endT = startT+datetime.timedelta(minutes=nMin)

    # Check if start/end times fall within file, if not truncate start/end to beginning/end of tAxis
    if startT < startDT:
        startIdx = 0
    else:
        tfromfilestart = (startT-startDT).seconds
        startIdx = int(tfromfilestart*sampFq)

    if endT > endDT:
        endIdx = -1
    else:
        tfromfileend = (endDT-endT).seconds
        endIdx = -int(tfromfileend*sampFq)

    subsetTime = tAxis[startIdx:endIdx]

    # Initialize empty subsetSignals Matrix:
    subsetSignals = np.zeros((len(elecs), len(subsetTime)))

    for ii, ee in enumerate(elecs):
        subsetSignals[ii] = signalArray[ee, startIdx:endIdx]

    return subsetSignals, subsetTime
