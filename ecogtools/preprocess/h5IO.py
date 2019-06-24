# 2015-07-06, LKirkby

import numpy as np
import pandas as pd
import datetime
import h5py
import matplotlib.dates as mdates
from scipy.stats import mstats


def read_hdf5_file(filepath):
    """
    Extracts signals and times from hdf5 file (does not save to another file)
    """

    try:
        fileobj = h5py.File(filepath, 'r')

    except:
        print 'Unable to open file'
        fileobj = 'NaN'
        signalArray = 'NaN'
        timeAxis = 'NaN'
        sampFq = 'NaN'

    # Run data extraction if open file successful:
    if fileobj != 'NaN':

        # Timing and sampling frequency:
        timeAxis = {}

        if 'timestamp vector' in fileobj:
            # UTS (unix time stamp): (seconds since 1970-01-01)
            timeAxis['UTC'] = fileobj['timestamp vector'].value
            nTimePoints = len(timeAxis['UTC'])

            if nTimePoints > 1:
                sampFq = round(
                    (nTimePoints)/(timeAxis['UTC'][-1]-timeAxis['UTC'][0]))
                nSec = nTimePoints/float(sampFq)
            else:

                sampFq = 'N/A'
                nSec = 1

            # Seconds since start of recording
            timeAxis['sec'] = np.linspace(0, nSec, nTimePoints)

            # matplotlib time axis (days since 0001-01-01 UTC plus 1)

            startObj = datetime.datetime.fromtimestamp(timeAxis['UTC'][0])
            endObj = datetime.datetime.fromtimestamp(timeAxis['UTC'][-1])

            mplStart = mdates.date2num(startObj)
            mplEnd = mdates.date2num(endObj)
            timeAxis['matplotlib'] = np.linspace(mplStart, mplEnd, nTimePoints)

        else:

            print 'No timestamp data in file'
            timeAxis = 'NaN'
            sampFq = 'NaN'

        # If sampling frequency is greater than 2048Hz (eg during stim, where it can be ~16kHz), split into shorter chunks

        fqCut = 2048

        nPts = fileobj['ECoG Array'].shape[0]
        nChans = fileobj['ECoG Array'].shape[1]

        signalArray_temp = np.zeros((nPts, nChans))
        signalArray = np.zeros_like(signalArray_temp).T

        if 'ECoG Array' in fileobj:

            if not isinstance(sampFq, str):

                if sampFq <= fqCut:

                    signalArray_temp = fileobj['ECoG Array'].value

                else:

                    iSplit = range(0, nPts, int(
                        np.ceil(nPts/float(sampFq/1000.))))
                    iSplit.append(nPts)

                    print('High sampFq, extracting data in chunks')
                    for ii, ind in enumerate(iSplit[:-1]):
                        #print('\t'+str(ii+1)+' out of '+str(len(iSplit)-1))
                        signalArray_temp[ind:iSplit[ii+1],
                                         :] = fileobj['ECoG Array'][ind:iSplit[ii+1], :]
            else:

                signalArray_temp = fileobj['ECoG Array'].value

            signalArray = signalArray_temp.T

        else:

            print 'No ECoG data in file'
            signalArray = 'NaN'

        fileobj.close()

    return {'signals': signalArray, 'timeAxis': timeAxis, 'sampFq': sampFq}


def zero_mean_signals(signalArray):

    signalArray_zeroMean = np.zeros_like(signalArray)

    if len(np.squeeze(signalArray).shape) > 1:
        for ss, signal in enumerate(signalArray):

            meanSig = np.mean(signal)
            signalMeaned = signal-meanSig
            signalArray_zeroMean[ss] = signalMeaned

    elif len(np.squeeze(signalArray).shape) == 1:

        meanSig = np.mean(signalArray)
        signalMeaned = signalArray-meanSig
        signalArray_zeroMean = signalMeaned

    return signalArray_zeroMean


def zscore_signals(signalArray):

    signalArray_zscore = np.zeros_like(signalArray)

    signalArray_zscore = mstats.zscore(signalArray)

    return signalArray_zscore


class MatlabH5File(h5py.File):
    """
    Class for decoding [<HDF5 object reference>] in .mat files
    """

    def decode(self, key):

        if self[key].dtype == 'float64':

            if self[key].shape == (1, 1):
                output = self[key][0][0]
            else:
                output = self[key].value.squeeze()

        elif self[key].dtype == 'uint16':
            output = ''
            for kk in self[key]:
                obj = kk[0]
                output = output+chr(obj)

        elif self[key].dtype == 'O':
            output = []
            for kk in self[key]:

                obj = self[kk[0]]
                string = ''.join(chr(oo) for oo in obj)
                output.append(string)
        else:
            print('Unknown dtype for {0}'.format(key))
            output = None

        return output


def cellarray_to_dataframe(h5object, key):
    """
    Convert <HDF5 object reference> (dtype('O')) to pandas dataframe, eg matlab cell array saved as v7.3 .mat file
    """

    all_df = []
    for column in h5object[key]:
        row_data = []
        for row_number in range(len(column)):
            firstElement = h5object[column[row_number]][:][0][0]
            if isinstance(firstElement, np.uint16):
                row_data.append(
                    ''.join(map(unichr, h5object[column[row_number]][:])))
            elif isinstance(firstElement, float):
                row_data.append(h5object[column[row_number]][:].squeeze())
            else:
                print('Unknown data type')

        all_df.append(row_data)

    return pd.DataFrame(all_df).T
