# 2015-07-06, LKirkby

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import collections
import os
import datetime
import glob
from sklearn.decomposition import PCA, FastICA, NMF
import scipy.io
import scipy.signal
import itertools

try:
    import cPickle as pickle
except:
    import pickle

try:
    import pyfftw.interfaces.numpy_fft as pfft
except:
    import numpy.fft as pfft

from ecogtools.tools import loaddata
from ecogtools.recordingparams import elecs as electrodeinfo


def calculate_coherence(signalArray, sampFq, tFrame=10, tSlide='None', tAxis='default', fqBands='default', PhaseRan=True, nRand=1):
    """
    Calculates signal coherence between all electrode pairs, over defined frequency bands
        - signalArray: np.array, ecog signals array (n electrodes x t timepoints)
        - sampFq: float, sampling frequency
        - tFrame: int, time bin in seconds for calculating coherence
        - tSlide: int or 'None', sliding window for moving coherence time bin
        - tAxis: np.array or 'default', time axis corresponding to signalArray time points
        - fqBands: dict eg fqBands['alpha'] = (8, 13) or 'default' to use pre-defined bands (in electrodeinfo.fqBands)
        - PhaseRan: bool. Perform phase randomization if True
        - nRand: number of phase randomizations to perform
    """

    nElectrodes = len(signalArray)

    if tSlide == 'None':
        tSlide = tFrame

    # Split up recording into time chuncks over which to calculate coherence
    tFrame = float(tFrame)
    tSlide = float(tSlide)

    # Select [:-1] in tAxis_split so that final time point has width >tFrame rather than <tFrame (cannot compute coherence if have too few time points)
    if tAxis == 'default':
        tDuration = np.floor(len(signalArray[0])/float(sampFq))
        tAxis = np.linspace(0, tDuration, len(signalArray[0]))
        tMax = tDuration-tFrame+tSlide
        tAxis_split = np.arange(0, tMax, tSlide)[:-1]
        tAxis_split_inds = tAxis_split[:-1]
    else:
        tDuration = np.floor(tAxis[-1]-tAxis[0])
        tMax = tDuration-tFrame+tSlide
        tAxis_split = np.arange(
            tAxis[0], (tAxis[-1]-tFrame+tSlide), tSlide)[:-1]
        tAxis_split_inds = np.arange(0, tMax, tSlide)[:-1]
    nTimepoints = len(tAxis_split)

    print('Recording duration: %0.0f sec (%0.2f mins)' %
          (tDuration, (tDuration/60.0)))
    print('Frame duration: %0.0f sec (%0.2f mins)' % (tFrame, (tFrame/60.0)))
    print('Sliding window: %0.0f sec (%0.2f mins)' % (tSlide, (tSlide/60.0)))
    print('Number of frames: %d\n' % (nTimepoints))

    # Define frequency bands
    if fqBands == 'default':
        fqBands = electrodeinfo.fq_bands()

    bands = fqBands.keys()

    # Initialize output matrices:
    Coherence = collections.OrderedDict()
    Phase = collections.OrderedDict()

    for band in bands:
        Coherence[band] = np.zeros((nElectrodes, nElectrodes, nTimepoints))

    for band in bands:
        Phase[band] = np.zeros((nElectrodes, nElectrodes, nTimepoints))

    # Define pairs over which to calculate coherences (list of tuples)
    ijPairs = []

    for i in range(nElectrodes):
        for j in range(i+1, nElectrodes):
            ijPairs.append((i, j))

    # Calculate coherence for those pairs
    # Loop through time index array:
    for nn, tt in enumerate(tAxis_split_inds[:-1]):

        if nn == 0:
            print('Loop '+str(nn+1)+' out of '+str(nTimepoints))
        elif nn % 250 == 0 and nn >= 250:
            print('Loop '+str(nn+1)+' out of '+str(nTimepoints))
        elif nn == (nTimepoints-1):
            print('Loop '+str(nn+1)+' out of '+str(nTimepoints))

        tStart = tt+tAxis[0]
        tEnd = tAxis_split_inds[nn+1]+tAxis[0]
        ttInds = np.where((tAxis >= tStart) & (tAxis < tEnd))[0]

        # Only perform coherence calculation if have > 1 second of data
        if len(ttInds) > int(sampFq):
            if (ttInds[-1]+1) <= len(tAxis):
                signalChunk = signalArray[0:nElectrodes,
                                          ttInds[0]:(ttInds[-1]+1)]
            else:
                signalChunk = signalArray[0:nElectrodes, ttInds[0]:]

            signalChunk = np.float32(signalChunk)

            Cxy, phase, fqs = mlab.cohere_pairs(
                signalChunk.T, ijPairs, Fs=sampFq, NFFT=int(sampFq/2))

            # Create numpy array of keys and values:
            ijCxyKeys = np.array(Cxy.keys())

            if PhaseRan == True:
                Coherence_surr = collections.OrderedDict()  # Create the empty dictionary
                for X in bands:
                    Coherence_surr[X] = np.zeros(
                        (nElectrodes, nElectrodes, nTimepoints))

                # Perform randomization several times (added: 25 July 2017)
                Cxy_surr_all = np.zeros(
                    (np.array(Cxy.values()).shape[0], np.array(Cxy.values()).shape[1], nRand))

                for nr in range(nRand):
                    signalChunk_surr = phaseran(signalChunk.T)
                    Cxy_surr, P0, F = mlab.cohere_pairs(
                        signalChunk_surr.T, ijPairs, Fs=sampFq, NFFT=int(sampFq/2))
                    Cxy_surr_all[:, :, nr] = np.array(Cxy_surr.values())

                Cxy_surr_mean = np.mean(Cxy_surr_all, axis=-1)
                ijCxyValues = np.array(np.squeeze(
                    Cxy.values())) - Cxy_surr_mean

            else:
                ijCxyValues = np.array(np.squeeze(Cxy.values()))

            ijphaseKeys = np.array(phase.keys())
            ijphaseValues = np.array(np.squeeze(phase.values()))

            # Extract frequency indices and calculate mean coherence over those bands:

            Indices = collections.OrderedDict()
            for X in bands:
                Indices[X] = list(
                    np.where((fqs >= fqBands[X][0]) & (fqs <= fqBands[X][1]))[0])

            if len(np.shape(np.squeeze(ijCxyValues))) == 2:

                MeansCoh = {}
                MeansPhase = {}
                for X in bands:
                    MeansCoh[X] = np.mean(np.squeeze(ijCxyValues)[
                                          :, Indices[X]], axis=1)
                    MeansPhase[X] = np.mean(np.squeeze(ijphaseValues)[
                                            :, Indices[X]], axis=1)

            elif len(np.shape(np.squeeze(ijCxyValues))) == 3:

                MeansCoh = {}
                MeansPhase = {}
                for X in bands:
                    MeansCoh[X] = np.mean(ijCxyValues[:, Indices[X]], axis=2)
                    MeansPhase[X] = np.mean(
                        ijphaseValues[:, Indices[X]], axis=2)

            # Fill coherence matrices:
            diagIndices = range(nElectrodes)
            for X in bands:
                # Set diagonals = 1
                Coherence[X][diagIndices, diagIndices, nn] = 1

                for pp, pair in enumerate(ijCxyKeys):
                    Coherence[X][pair[0], pair[1], nn] = MeansCoh[X][pp]
                    Coherence[X][pair[1], pair[0], nn] = MeansCoh[X][pp]

            # Fill phase matrices:
            diagIndices = range(nElectrodes)
            for X in bands:
                # Set diagonals = 1
                Phase[X][diagIndices, diagIndices, nn] = 1

                for pp, pair in enumerate(ijphaseKeys):
                    Phase[X][pair[0], pair[1], nn] = MeansPhase[X][pp]
                    Phase[X][pair[1], pair[0], nn] = MeansPhase[X][pp]

        # If no data for that timepoint:
        elif len(ttInds) == 0:

            # Fill coherence matrices:
            diagIndices = range(nElectrodes)
            for X in bands:
                # Set diagonals = 1
                Coherence[X][diagIndices, diagIndices, nn] = np.nan

                for pp, pair in enumerate(ijCxyKeys):
                    Coherence[X][pair[0], pair[1], nn] = np.nan
                    Coherence[X][pair[1], pair[0], nn] = np.nan

            # Fill phase matrices:
            diagIndices = range(nElectrodes)
            for X in bands:
                # Set diagonals = 1
                Phase[X][diagIndices, diagIndices, nn] = np.nan

                for pp, pair in enumerate(ijphaseKeys):
                    Phase[X][pair[0], pair[1], nn] = np.nan
                    Phase[X][pair[1], pair[0], nn] = np.nan

    # Remove Nans and infs:

    for X in bands:
        Coherence[X] = np.nan_to_num(Coherence[X])
        Phase[X] = np.nan_to_num(Phase[X])

    print('Done\n')

    return {'Coherence': Coherence, 'Phase': Phase, 'deltaT': tFrame, 'tAxis': tAxis_split}


def phaseran(signal):
    """
    Performs phase randomization for coherence matrices.

    NOTE: Signal input has to be nTimepoints x nElectrodes (there is a check)
    """

    # check that it is in right orientation
    if signal.shape[1] > signal.shape[0]:
        signal = signal.T

    # Get parameters
    nTimepoints = signal.shape[0]
    nElectrodes = signal.shape[1]

    # Check to make sure that it is an odd number of samples
    if nTimepoints % 2 == 0:
        signal = signal[:-1, :]
        nTimepoints = nTimepoints-1

    nTimepoints = signal.shape[0]
    len_ser = (nTimepoints-1)/2
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, nTimepoints)

    # fft_A = pfft.builders.fft(signal, axis=0) # FFT of original data
    try:
        fft_A = pfft.fft(signal, axis=0, threads=15)
    except:
        fft_A = pfft.fft(signal, axis=0)

    # Create the random phases for all the time series
    ph_rnd = np.random.rand(len_ser, nElectrodes)
    ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
    ph_interv2 = np.conj(np.flipud(ph_interv1))

    # Randomize all time series simultaneously
    fft_recblk_surr = fft_A
    fft_recblk_surr[interv1, :] = fft_A[interv1, :]*ph_interv1
    fft_recblk_surr[interv2, :] = fft_A[interv2, :]*ph_interv2
    surrblk = np.float32(pfft.ifft(fft_recblk_surr, axis=0)).T

    return surrblk


def ave_brainregions(cohInputDict, brainRegions):
    """
    Reduces dimentionality of electrode-by-electrode coherence matrices by averaging across common brain regions
    Input is dictionary output from calculate_coherences
    """

    cohData_coherence = collections.OrderedDict()
    cohData_phase = collections.OrderedDict()

    for band in cohInputDict['Coherence'].keys():
        cohData_coherence[band] = average_coh_matrices(
            cohInputDict['Coherence'][band], brainRegions)
        cohData_phase[band] = average_coh_matrices(
            cohInputDict['Phase'][band], brainRegions)

    return {'Coherence': cohData_coherence, 'Phase': cohData_phase, 'deltaT': cohInputDict['deltaT'], 'tAxis': cohInputDict['tAxis']}


def average_coh_matrices(cohData, brainRegions):
    """
    Computes and returns coherence matrices averaged over common brain regions
    """

    refElectrodes = brainRegions.values()
    refNames = brainRegions.keys()

    nRegions = len(refNames)

    # Assign number to each brainRegion:
    refNumber = range(nRegions)

    # Define pairs over which to average coherences (list of tuples)
    ijPairs = []
    for i in range(nRegions):
        for j in range(i+1, nRegions):
            ijPairs.append((i, j))

    # Define output matrix:
    nTimePoints = cohData.shape[-1]
    cohAve = np.zeros((nRegions, nRegions, nTimePoints))

    n = 0
    nPrintList = range(0, nTimePoints, 1000)
    nPrintList.append(nTimePoints)

    # Check for nans: replace all nans with zeros and infs with finite numbers
    cohData = np.nan_to_num(cohData)

    # Loop over time points and pairs to get averages:
    for nn in range(nTimePoints):
        if nTimePoints > 1000:
            if nn == nPrintList[n]:
                #print('t = '+str(nn)+'/'+str(nTimePoints))
                n += 1
        matrix = cohData[:, :, nn]
        for pair in ijPairs:
            electrodesA = refElectrodes[pair[0]]
            electrodesB = refElectrodes[pair[1]]
            firstElectrodeA = electrodesA[0]
            lastElectrodeA = electrodesA[-1]+1
            firstElectrodeB = electrodesB[0]
            lastElectrodeB = electrodesB[-1]+1

            meanCoherence = np.mean(
                matrix[firstElectrodeA:lastElectrodeA, firstElectrodeB:lastElectrodeB])

            cohAve[pair[0], pair[1], nn] = meanCoherence
            cohAve[pair[1], pair[0], nn] = meanCoherence

        # Fill in diagonals:
        for diagonal in refNumber:
            electrodes = refElectrodes[diagonal]
            firstElectrode = electrodes[0]
            lastElectrode = electrodes[-1]+1

            meanCoherence = np.mean(
                matrix[firstElectrode:lastElectrode, firstElectrode:lastElectrode])

            cohAve[diagonal, diagonal, nn] = meanCoherence

    return cohAve


def dimred_pca_ica(dataIn, nCs=10, calcPC=True, calcIC=True, sparsePCA=False):
    """
    Performs PCA and ICA on coherence matrices. Option for sparse PCA calculation

    Input can be either cohData dictionary or path to cohData file
    """

    if isinstance(dataIn, str):
        cohFile = dataIn
        cohData = loaddata.reload_coherence(cohFile)

        saveFlag = True
        PCsavefile = cohFile.split('.')[0]+'_PCA'
        ICsavefile = cohFile.split('.')[0]+'_ICA'

    elif isinstance(dataIn, dict):
        cohData = dataIn
        saveFlag = False

    if isinstance(cohData, dict):
        data = cohData['Coherence']

        pc = collections.OrderedDict()
        pcEigV = collections.OrderedDict()
        ic = collections.OrderedDict()

        if calcPC == True:
            print('PCA -')
            for band in data.keys():
                print('       '+str(band))

                if sparsePCA == True:
                    pc[band], pcEigV[band] = sparse_pca_calculation(
                        data[band], nCs)

                else:
                    pc[band], pcEigV[band] = pca_calculation(data[band], nCs)

                    # Percent variance explained by each PC
                    plt.plot(pcEigV[band])
                    plt.title('Percent variance explained by PCA')
                    plt.ylabel('Fraction')
                    plt.xlabel('PC')

            plt.show()

            pc['fileID'] = cohData['fileID']

            # Save if filename input:
            if saveFlag == True:
                handle = open(PCsavefile+'.p', 'wb')
                pickle.dump({'pc': pc, 'pcEigV': pcEigV}, handle, protocol=2)
                handle.close()

        if calcIC == True:
            print('ICA -')
            for band in data.keys():
                ic[band] = ica_calculation(data[band], nCs)
                print('       '+str(band))

            ic['fileID'] = cohData['fileID']

            # Save if filename input:
            if saveFlag == True:
                # Save parameters:
                handle = open(ICsavefile+'.p', 'wb')
                pickle.dump({'ic': pc}, handle, protocol=2)
                handle.close()

    elif isinstance(cohData, np.ndarray):

        if calcPC == True:
            print('PCA -')
            pc, pcEigV = pca_calculation(cohData, nCs)
            np.save(PCsavefile, pc)
#            scipy.io.savemat(PCsavefile, {'PCA': pc})

            # Percent variance explained by each PC
            plt.plot(pcEigV)
            plt.title('Percent variance explained by PCA')
            plt.ylabel('Fraction')
            plt.xlabel('PC')
            plt.show()

        if calcIC == True:
            print('ICA -')
            ic = ica_calculation(cohData, nCs)
            np.save(ICsavefile, ic)

    if calcPC == False:
        pc = 'None'
    if calcIC == False:
        ic = 'None'

    return pc, ic


def pca_calculation(dataMatrixIn, nPCs, meanSubtract=True, useHalfMatrix=True):
    """
    Single PCA calculation of coherence matrices

    - dataMatrixIn: time series of analysis matrix (eg nElectrodes*nElecrodes*tTimepoints)
    - nPCs: number of PCs to calculate
    - meanSubtract: True or False. Subtracts mean across time if true
    - useHalfMatrix: True or False. If True (eg. if matrix is symmetric) then perform PCA on half of matrix only
    """

    dataMatrix0 = np.zeros_like(dataMatrixIn)
    dataMatrix = np.zeros_like(dataMatrixIn)

    # Mean subtraction:
    if meanSubtract == True:
        print('\tMean subtracting')
        meanCoh = np.mean(dataMatrixIn, axis=-1)
        for cc in np.arange(dataMatrixIn.shape[-1]):
            dataMatrix0[:, :, cc] = dataMatrixIn[:, :, cc] - meanCoh
    elif meanSubtract == False:
        dataMatrix0 = dataMatrixIn

    # Half or all matrix:
    if useHalfMatrix == False:
        dataMatrix = dataMatrix0
    elif useHalfMatrix == True:
        print('\tGetting half matrix')
        for ii in range(dataMatrix0.shape[0]):
            for jj in range(ii, dataMatrix0.shape[0]):
                dataMatrix[ii, jj, :] = dataMatrix0[ii, jj, :]

    # PCA calculation:
    pca = PCA(n_components=nPCs)
    sz = dataMatrix.shape

    X = np.reshape(dataMatrix, (sz[0]*sz[1], sz[2]))
    # X = X.T # Form needs to be nDimension x T
    print('\tCalculating pca')
    # Reconstruct signals
    pc = np.reshape(pca.fit_transform(X), (sz[0], sz[1], nPCs))
    pcEigV = pca.explained_variance_ratio_

    # Fill in symmetrical matrix if ran PCA on half:
    if useHalfMatrix == True:
        print('Getting full matrix')
        for ii in range(dataMatrix.shape[0]):
            for jj in range(ii+1, dataMatrix.shape[0]):
                pc[jj, ii, :] = pc[ii, jj, :]

    return pc, pcEigV


def ica_calculation(dataMatrixIn, nICs, useHalfMatrix=True, **kwargs):
    """
    Single ICA calculation of coherence matrices
    - dataMatrixIn: time series of analysis matrix (eg nElectrodes*nElecrodes*tTimepoints)
    - nICs: number of ICs to calculate
    - useHalfMatrix: True or False. If True (eg. if matrix is symmetric) then perform ICA on half of matrix only
    """

    dataMatrix0 = dataMatrixIn
    dataMatrix = np.zeros_like(dataMatrixIn)

    # Half or full matrix:
    if useHalfMatrix == False:
        dataMatrix = dataMatrix0
    elif useHalfMatrix == True:
        for ii in range(dataMatrix0.shape[0]):
            for jj in range(ii, dataMatrix0.shape[0]):
                dataMatrix[ii, jj, :] = dataMatrix0[ii, jj, :]

    # ICA
    if kwargs and 'max_iter' in kwargs:
        ica = FastICA(n_components=nICs,
                      max_iter=kwargs['max_iter'], tol=0.001)
    else:
        ica = FastICA(n_components=nICs, max_iter=2000, tol=0.001)
    sz = dataMatrix.shape
    X = np.reshape(dataMatrix, (sz[0]*sz[1], sz[2]))
    # X = X.T # Form needs to be nDimension x T

    # Adding in exception in case data does not converge
    try:
        # Reconstruct signals
        ic = np.reshape(ica.fit_transform(X), (sz[0], sz[1], nICs))

        # Fill in symmetrical matrix if ran ICA on half:
        if useHalfMatrix == True:
            for ii in range(dataMatrix.shape[0]):
                for jj in range(ii+1, dataMatrix.shape[0]):
                    ic[jj, ii, :] = ic[ii, jj, :]
    except:
        print('ICA did not converge')

    return ic


def project_onto_component(dataMatrix, compMatrix, symmMat=True):
    """
    Project data (eg coherence matrices) onto components (PCs or ICs)

    - dataMatrix: nElectrodes*nElectrodes*tTimepoints
    - compMarix: nElectrodes*nElectrodes*nComponents
    - norm: True or False. Normalize projection if True

    Returns
    - projMatrix: tTimepoints*nComponents; projection onto each component
    """

    # Determine number of components and number of time points
    if len(np.shape(compMatrix)) == 3:
        nRegions = np.shape(compMatrix)[0]
        nCpts = np.shape(compMatrix)[-1]
    elif len(np.shape(compMatrix)) == 2:
        nRegions = np.shape(compMatrix)[0]
        nCpts = 1
        compMatrix = np.atleast_3d(compMatrix)

    tPts = dataMatrix.shape[-1]

    projMatrix = np.zeros((tPts, nCpts))

    dataMatrix0 = np.zeros_like(dataMatrix)
    compMatrix0 = np.zeros_like(compMatrix)

    # If matrix is symmetrical take only half data and comp matrices so as not to double count off diagonal regions
    if symmMat == True:
        for ii in range(dataMatrix.shape[0]):
            for jj in range(ii, dataMatrix.shape[0]):
                dataMatrix0[ii, jj, :] = dataMatrix[ii, jj, :]
                compMatrix0[ii, jj, :] = compMatrix[ii, jj, :]
    elif symmMat == False:
        dataMatrix0 = dataMatrix
        compMatrix0 = compMatrix

    dataLinear = np.reshape(dataMatrix0, (nRegions*nRegions, tPts))
    compLinear = np.reshape(compMatrix0, (nRegions*nRegions, nCpts))

    projMatrix = np.dot(compLinear.T, dataLinear)

    return projMatrix


def combine_coherence_files(input_dir, coh_str, fileExtension='.p', excludeGrid=False):
    """
    Combines coherence matrices from different pickle files (associated with different edf files for the same patient)
    Saves output to npy files according to frequency band
    Also saves associated time axis to npy file

    Inputs:
    input_dir: directory with pickle coherence dictionary files
    coh_str: common string in filenames across files of interest, eg 'coherence_30s_pr'
    """

    # Combine files with coh_str in filename:
    cohFiles = glob.glob(input_dir+os.sep+'*'+coh_str+'*'+fileExtension)

    # Input files
    all_times = []
    coh_allT = {}
    phase_allT = {}

    kk = 0
    for cohFile in cohFiles:

        print 'Loading file ' + \
            str(kk+1) + ' out of ' + str(len(list(cohFiles)))

        cohData = np.load(cohFile)

        if kk == 0:
            all_times = cohData['tAxis']
        else:
            all_times = np.concatenate((all_times, cohData['tAxis']), axis=0)

        Coherence = cohData['Coherence']
        Phase = cohData['Phase']
        print 'n time points = '+str(Coherence['theta'].shape[-1])
        # concatenate all the bands

        if excludeGrid == False:

            for band in Coherence.keys():
                if kk == 0:  # assign matrices on first case
                    coh_allT[band] = Coherence[band]
                    phase_allT[band] = Phase[band]
                else:
                    coh_allT[band] = np.concatenate(
                        (coh_allT[band], Coherence[band]), axis=-1)
                    phase_allT[band] = np.concatenate(
                        (phase_allT[band], Phase[band]), axis=-1)

        elif excludeGrid == True:

            print 'Excluding 64-electrode grid'
            for band in Coherence.keys():
                if kk == 0:  # assign matrices on first case
                    coh_allT[band] = Coherence[band][64:, 64:, :]
                    phase_allT[band] = Phase[band][64:, 64:, :]
                else:
                    coh_allT[band] = np.concatenate(
                        (coh_allT[band], Coherence[band][64:, 64:, :]), axis=-1)
                    phase_allT[band] = np.concatenate(
                        (phase_allT[band], Phase[band][64:, 64:, :]), axis=-1)

        kk += 1

    coh_allT_unique = {}
    phase_allT_unique = {}

    # Sort the output: (unique also sorts indices)
    all_times_unique, uniqueIndices = np.unique(all_times, return_index=True)
    for band in Coherence.keys():
        coh_allT_unique[band] = coh_allT[band][:, :, uniqueIndices]

    for band in Phase.keys():
        phase_allT_unique[band] = phase_allT[band][:, :, uniqueIndices]

    print '\tCombining '+coh_str+' files done\n'

    return coh_allT_unique, phase_allT_unique, all_times_unique


def is_grid_removed(patientID, dataMatrix):
    """
    Check if coherence matrix has 64-electrode lateral-frontotemporal grid (abbr: lFTC_G)

    Returns True or False
    """

    nElecsAll = electrodeinfo.electrode_locs(patientID)['nElectrodes']
    nElecsMat = np.shape(dataMatrix)[0]

    if nElecsAll-nElecsMat == 64:
        noGrid = True
    elif nElecsMat == nElecsAll:
        noGrid = False
    else:
        print 'Electrode number mismatch'
        noGrid = None

    return noGrid


def subset_matrix(dataMatrix, subsetInds):
    """
    Extract subset of dataMatrix according to subsetInds.

        - dataMatrix: 2-D or 3-D matrix eg. could be time-series of coherence matrices
        - subsetInds: list of indices of interest

    NB. good to check if dataMatrix has 64-electrode grid first using is_grid_removed function so that subsetInds align correctly with dataMatrix.shape

    ## See below for get_coherence_subset() function##
    """

    # Initialize subset matrix
    nElecs = len(subsetInds)
    if len(dataMatrix.shape) == 3:
        subsetMatrix = np.zeros((nElecs, nElecs, dataMatrix.shape[-1]))

    elif len(dataMatrix.shape) == 2:
        subsetMatrix = np.zeros((nElecs, nElecs))
        dataMatrix = np.atleast_3d(dataMatrix)

    # Fill matrix
    print 'Extracting subset matrix'
    for ii, ss in enumerate(subsetInds):
        if (ii > 0) and (ii % 10 == 0):
            print '\t'+str(ii)+' of '+str(len(subsetInds))+' done'
        subsetMatrix[ii, ii:, :] = dataMatrix[ss, subsetInds[ii:], :]
        subsetMatrix[ii:, ii, :] = subsetMatrix[ii, ii:, :]

    return subsetMatrix


def get_region_subset(patient, brainRegs, band, dataType='coherence', tFrame=10, excludeGrids=True, matType='full', **kwargs):
    """
    Get subset of coherence/phase matrix corresponding to brainRegs

    Inputs:
        - patient: object of Patient class
        - brainRegs: list of strings of brain region abbreviations eg ['AM', 'HPC'] etc
        - band: string of frequency band eg 'beta'
        - dataType: can be 'coherence' or 'phase'
        - tFrame: time interval for coherence matrix time series
        - excludeGrids: boolean. If True then do not include electrodes located on grid (as opposed to strips/depths)
        - matType: can be 'full' (all electrodes) or 'average' (average across electrodes)
    """

    if dataType == 'coherence':
        dataOut = patient.loadCoherence(band, tFrame, matType=matType)
    elif dataType == 'phase':
        dataOut = patient.loadPhase(band, tFrame, matType=matType)
    data_all = dataOut[dataType]
    tAxis = dataOut['tAxis']

    if matType == 'full':
        # Check if 64-electrode grid is removed from coherence matrix
        no64Grid = is_grid_removed(patient.getID(), data_all)

        # Get subset matrix
        regs, regsNew = electrodeinfo.choose_electrodes(
            patient.ID, brainRegs, no64Grid=no64Grid, excludeGrids=excludeGrids)
        subsetInds = list(itertools.chain.from_iterable(regs.values()))
    elif matType == 'average':
        regs = electrodeinfo.get_region_index(patient.ID, brainRegs)
        subsetInds = regs.values()

        regsNew = collections.OrderedDict()
        for kk, key in enumerate(regs):
            regsNew[key] = kk

    data_subset = subset_matrix(data_all, subsetInds)

    outputDict = {}
    outputDict[dataType] = data_subset
    outputDict['tAxis'] = tAxis
    outputDict['band'] = band
    outputDict['regs'] = regs
    outputDict['regsNew'] = regsNew

    return outputDict


def find_ICs_from_PCA(dataMatrix, nPCs=50, meanSubtract=True, useHalfMatrix=True, method='ICA', **kwargs):
    """
    Find independent components based on number of significant PCs using marcenko-pastur limit

    Inputs:
        - method: matrix decomposition method for calculating ICNs, eg 'ICA' (independent components analysis) -- other methods to be added (eg. NMF)
    """

    print 'Running PCA'
    PCs, eigVs = pca_calculation(
        dataMatrix, nPCs=nPCs, useHalfMatrix=useHalfMatrix, meanSubtract=meanSubtract)

    eigVthresh = eigenval_thresh(dataMatrix)

    nSig = len(np.where(eigVs >= eigVthresh)[0])

    if method.lower() == 'ica':
        print 'Running ICA'
        ICs = ica_calculation(
            PCs[:, :, :nSig], nICs=nSig, useHalfMatrix=useHalfMatrix, **kwargs)

    else:
        print('Matrix decomposition method not understood')
        ICs = None

    return ICs


def eigenval_thresh(dataMatrix):
    """    
    Eigenvalue threshold, the number above which can be used to estimate the number of statistically significant components in the data matrix.
    """

    sz = dataMatrix.shape
    if len(sz) == 3:
        dataMatrix_2d = dataMatrix.reshape((sz[0]*sz[1], sz[2]))
    else:
        dataMatrix_2d = dataMatrix

    nRows = dataMatrix_2d.shape[0]
    nCols = dataMatrix_2d.shape[1]

    sigma2 = np.mean(np.var(dataMatrix_2d, axis=1))

    lam_max, lam_min = marcenko_pastur(nRows, nCols, sigma2)

    return lam_max


def marcenko_pastur(nRows, nCols, sigma2):
    """
    Finds theoretical boundary of eigenvalues for correlation matrix of a normal random matrix M with statistically independent rows

    Inputs:
        - nCols: number of columns in M
        - nRows: number of rows in M
        - sigma2: variance of elements in M

    Refer to Lopes-dos-Santos et al., 2013. Marcenko and Pastur (1967) showed that the eigenvalues of the correlation matrix of a
    normal random matrix M with statistically independent rows follow a probability function (quadratic) with roots given by lambda_max, lambda_min, below
    """
    q = float(nRows)/float(nCols)

    lambda_max = sigma2*(1+np.sqrt(q))**2
    lambda_min = sigma2*(1-np.sqrt(q))**2

    return lambda_max, lambda_min


def find_ICNs(patient, brainRegs, band, tFrame=10, method='ICA', excludeGrid=True, matType='full', **kwargs):
    """
    Find intrinsic connectivity networks (ICNs) from coherence dataset

    Inputs:
        - patient: object of Patient class
        - brainRegs: list of strings of brain region abbreviations eg ['AM', 'HPC'] etc
        - band: string of frequency band eg 'beta'
        - tFrame: time interval for coherenc ematrix time series
        - method: matrix decomposition method for calculating ICNs, eg 'ICA' (independent components analysis) -- other methods to be added (eg. NMF)
    """

    cohsSubset = get_region_subset(patient, brainRegs, band, dataType='coherence',
                                   tFrame=tFrame, excludeGrids=excludeGrid, matType=matType, **kwargs)

    # Find ICNs and projections
    icnMat = find_ICs_from_PCA(
        cohsSubset['coherence'], method=method, **kwargs)
    icnProj = project_onto_component(
        cohsSubset['coherence'], icnMat, symmMat=True)

    return {'Cohs': cohsSubset['coherence'], 'ICNs': icnMat, 'Projections': icnProj, 'tAxis': cohsSubset['tAxis'], 'Band': cohsSubset['band'], 'Regs': cohsSubset['regsNew']}
