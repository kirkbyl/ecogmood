# 2015-07-06, LKirkby
# 2015-07-16: edited to remove need for 'header' input

import numpy as np
import matplotlib.pyplot as plt
import math

from scipy import signal
from scipy.signal.filter_design import cheby1
from scipy.signal.fir_filter_design import firwin
import scipy.signal.signaltools as sigtool

def filter_ecog_signal(signalArray, sampFq, filtFqs=[60,120,180], filtBandwidths=[4,4,4], filtOrder=2, plotSpectrogram=False, plotFilter=False):    
    """
    Performs notch filtering of input frequencies (filtFqs with bandwidth filtBandwidths)
    Option to plot spectrogram and filter
    Returns filtered signals
    """
            
    NyquistFq = float(sampFq/2)
    filtRanges = [[0 for j in range(2)] for i in range(len(filtFqs))]
    
    for p in range(len(filtRanges)):
        filtRanges[p][0] = (filtFqs[p] - filtBandwidths[p]/2)/NyquistFq
        filtRanges[p][1] = (filtFqs[p] + filtBandwidths[p]/2)/NyquistFq   

    signalArrayFiltered = np.zeros_like(signalArray)
    
    # Check if signalArray is multidimensional, or just corresponds to a single electrode:
    if len(np.shape(signalArray)) > 1:
        nElectrode = 0
        for electrode in signalArray:
            for filtRange in filtRanges:
                if filtRange[1]<1:
                    filtNume, filtDenom = signal.butter(filtOrder, filtRange, btype='bandstop')
                    electrode = signal.filtfilt(filtNume, filtDenom, electrode)
                    if plotFilter:
                        # Only want to plot filter once in the loop:
                        if nElectrode==0:
                            w, h = signal.freqs(filtNume, filtDenom, 1000)
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.plot(w, 20 * np.log10(abs(h)))
                            ax.set_xscale('log')
                            ax.set_title('Filter frequency response')
                            ax.set_xlabel('Frequency [radians / second]')
                            ax.set_ylabel('Amplitude [dB]')
                            #ax.axis((10, 1000, -100, 10))
                            ax.grid(which='both', axis='both')
                            plt.show()
            if plotSpectrogram:
                fig = plt.figure()
                plt.specgram(electrode, NFFT=512, Fs=sampFq)
                plt.title('Filtered spectrogram for electrode '+str(nElectrode))
                plt.xlabel('Time [sec]')
                plt.ylabel('Frequency [Hz]')
#                plt.xlim(0, (header['edfInfo']['nRecords']-2))
                plt.ylim(0, sampFq/2.)
                cbar = plt.colorbar()
                cbar.set_label('Power spectral density [W/Hz]') 
                plt.show()
            signalArrayFiltered[nElectrode] = electrode
            nElectrode=nElectrode+1
    # if signalArray is one dimensional vector (single electrode)
    elif len(np.shape(signalArray)) == 1:
        nElectrode = 0
        electrode = signalArray
        for filtRange in filtRanges:
            if filtRange[1]<1:
                filtNume, filtDenom = signal.butter(filtOrder, filtRange, btype='bandstop')
                electrode = signal.filtfilt(filtNume, filtDenom, electrode)
                if plotFilter:
                    w, h = signal.freqs(filtNume, filtDenom, 1000)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(w, 20 * np.log10(abs(h)))
                    ax.set_xscale('log')
                    ax.set_title('Filter frequency response')
                    ax.set_xlabel('Frequency [radians / second]')
                    ax.set_ylabel('Amplitude [dB]')
                    #ax.axis((10, 1000, -100, 10))
                    ax.grid(which='both', axis='both')
                    plt.show()
        if plotSpectrogram:
            fig = plt.figure()
            plt.specgram(electrode, NFFT=512, Fs=sampFq)
            plt.title('Filtered spectrogram for electrode '+str(nElectrode))
            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [Hz]')
#            plt.xlim(0, (header['edfInfo']['nRecords']-2))
            plt.ylim(0, sampFq/2.)
            cbar = plt.colorbar()
            cbar.set_label('Power spectral density [W/Hz]') 
            plt.show()
        signalArrayFiltered = electrode

    return signalArrayFiltered

   
def common_average_reference(signalArray, refRegions='all', method='mean'):
    """
    Performs common average referencing (spatial filtering)
    refRegions: list of lists of electrode numbers. Default 'all': all regions/electrodes. [Normally I do: refRegions = brainRegions.values() ie. list of lists of elec values for each lead/probe]
    method: 'mean' or 'median'
    
    Returns CARed signals
    """
    
    carSignal = np.zeros_like(signalArray)
    
    if refRegions == 'all':
        nElecs = signalArray.shape[0]
        refRegions = [range(nElecs)]
    
    for electrodes in refRegions:
        
        # Do not CAR if only have 1 or 2 elecs (would be subtracting too much signal)
        if len(electrodes) > 2:
            if method == 'mean':
                meanSignal = np.mean(signalArray[electrodes,:], axis=0)
    
            elif method == 'median':
                meanSignal = np.median(signalArray[electrodes,:], axis=0)            
                
            for electrode in electrodes:
                carSignal[electrode][:] = signalArray[electrode][:] - meanSignal
                
        else:
            carSignal[electrode][:] = signalArray[electrode][:]
        
    return carSignal



# Generic band pass/stop, high/low pass filters:

def butter_bandpass_filter(data, sampFq, lowcut, highcut, order=5, btype='bandpass'):
    """
    btype can be 'bandpass' or 'bandstop'
    """
    
    b, a = butter_bandpass(lowcut, highcut, sampFq, order, btype=btype)
    filtsig = signal.filtfilt(b, a, data)
    
    return filtsig
    

def butter_highpass_filter(data, sampFq, cutoff, order=5):

    b, a = butter_cutoff(cutoff, sampFq, order, btype='highpass')
    filtsig = signal.filtfilt(b, a, data)
    
    return filtsig    
    
    
def butter_lowpass_filter(data, sampFq, cutoff, order=5):

    b, a = butter_cutoff(cutoff, sampFq, order, btype='lowpass')
    filtsig = signal.filtfilt(b, a, data)
    
    return filtsig    
        

def butter_bandpass(lowcut, highcut, fs, order, btype):
    
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    
    b, a = signal.butter(order, [low, high], btype=btype)
    
    return b, a


def butter_cutoff(cutoff, fs, order, btype):
    
    nyq = fs*0.5
    cut = cutoff/nyq
    
    b, a = signal.butter(order, cut, btype=btype)
    
    return b, a

    
def downsample_signals(signalArray, sampFq, sampFqTarget, tAxis):
    """
    Downsample signal sampled at sampFq to sampFqTarget
    """
    
    # Round up to nearest 100 for cutoff:
    fqCut = int(math.ceil(sampFqTarget/100.))*100
    
    # Decimation factor:  
    Dfactor = int(round(sampFq/sampFqTarget))
    sampFq_deci = sampFq/float(Dfactor)
    
    if (sampFq > fqCut) & (Dfactor > 1):
            
        # Bandpass signals first: To avoid decimated output aliasing errors, we must satisfy
        # the Nyquist criterion and ensure that signals_bandpass's bandwidth B is not greater than sampFq_deci/2
                                        
        signals_bandpass = np.zeros_like(signalArray)
        Nq_deci = sampFq_deci/2.
        
        print('Decimating signals, resampling to %dHz (sampFq = %.2fkHz)...' % (sampFq_deci, (sampFq/1000.)))
        print('Lowpass filtering to avoid aliasing (<%dHz)...' % Nq_deci)
        
        for ee, electrodesig in enumerate(signalArray):
            
            filtsig = butter_lowpass_filter(electrodesig, sampFq_deci, Nq_deci, order=2)
            signals_bandpass[ee] = filtsig 
                                        
    
        # Decimate signals by using a filter (modified from scipy.signal.decimate) 
        signals_deci = np.zeros((signalArray.shape[0], signalArray.shape[-1]/Dfactor))
    
    
        for ee, electrodesig in enumerate(signals_bandpass):
            
            decisig = decimate_filtfilt(electrodesig, Dfactor, ftype='fir', zerophase=True)
            
            if ee == 0:
                if signals_deci.shape[0] != len(decisig):
                    signals_deci = np.zeros((signalArray.shape[0], len(decisig)))
            
            signals_deci[ee] = decisig
            
        # New time axis
        tAxis_deci = np.linspace(tAxis[0], tAxis[-1], signals_deci.shape[-1])
        
    else:
        
        print('Not decimating signals (sampFq = %.2fkHz)...' % (sampFq/1000.))
        sampFq_deci = sampFq
        tAxis_deci = tAxis
        signals_deci = signalArray  
        
        
    return signals_deci, sampFq_deci, tAxis_deci
        
         
    
# Modification of scipy.signal.decimate function: introduced forward/backward filtering (with filtfilt)
# to remove phase shift introduced by single low pass filter (lfilter)

def decimate_filtfilt(x, q, n=None, ftype='iir', axis=-1, zerophase=True):
    """
    Downsample the signal by using a filter.

    By default, an order 8 Chebyshev type I filter is used.  A 30 point FIR
    filter with hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : ndarray
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.
    n : int, optional
        The order of the filter (1 less than the length for 'fir').
    ftype : str {'iir', 'fir'}, optional
        The type of the lowpass filter.
    axis : int, optional
        The axis along which to decimate.

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    See also
    --------
    resample

    """

    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        if ftype == 'fir':
            n = 30
        else:
            n = 8

    if ftype == 'fir':
        b = firwin(n + 1, 1. / q, window='hamming')
        a = 1.
    else:
        b, a = cheby1(n, 0.05, 0.8 / q)

    if zerophase == True:
        y = signal.filtfilt(b, a, x, axis=axis)
    elif zerophase == False:
        y = signal.lfilter(b, a, x, axis=axis)
        

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[sl]
