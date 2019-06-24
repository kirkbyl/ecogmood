# 2015-07-06, LKirkby

from ecogtools.tools import loaddata, utilities as utils
from ecogtools.recordingparams import elecs as electrodeinfo
from scipy.stats.stats import pearsonr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')


def plot_matrices(data, tAxis=None, subset=range(10), dataType='Coherence', tStart=(), labels=None, title='', **kwargs):

    if len(data.shape) == 3:
        if data.shape[-1] >= len(subset):
            subset = subset
        else:
            subset = range(data.shape[-1])
    elif len(data.shape) == 2:
        subset = range(1)
        data = np.atleast_3d(data)

    if tAxis is not None:
        if isinstance(tAxis[0], datetime.datetime) == False:
            tAxisStr = utils.convert_timestamp(tAxis)['datetime']

        if tStart != ():
            tStartDT = utils.convert_timestamp(
                tStart, isTuple=True)['datetime']

            iiStart = np.where(np.array(tAxis) >= tStartDT)[0][0]
            subset = [ss+iiStart for ss in subset]

    fig, ax, ax1D = utils.custom_subplots(len(subset), colorbar=True)

    for kk, X in enumerate(subset):

        if dataType == 'PCA':
            ax1D[kk].set_title('PC '+str(X+1))
        elif dataType == 'ICA':
            ax1D[kk].set_title('IC '+str(X+1))
        else:
            try:
                ax1D[kk].set_title(tAxisStr[X].strftime(
                    '%Y-%m-%d, %H:%M:%S'), fontsize=11)
            except:
                ax1D[kk].set_title('')
        plot_matrix_single(data[:, :, X], dataType,
                           fig=fig, ax=ax1D[kk], **kwargs)

        if labels != None and data.shape[0] == len(labels):
            if X == subset[0]:
                ax1D[kk].set_yticks(np.arange(len(labels)))
                ax1D[kk].set_yticklabels(labels)
                ax1D[kk].set_xticks([], '')
            elif X == subset[-1]:
                ax1D[kk].set_yticks([], '')
                ax1D[kk].set_xticks(np.arange(len(labels)))
                ax1D[kk].set_xticklabels(labels, rotation=90)
            elif X != subset[0] or X != subset[-1]:
                ax1D[kk].set_yticks([], '')
                ax1D[kk].set_xticks([], '')
        else:
            ax1D[kk].set_yticks([], '')
            ax1D[kk].set_xticks([], '')

    plt.suptitle(title, fontsize=14)
    plt.show()

    return fig


def plot_matrix_single(data, dataType=None, ax=None, fig=None, **kwargs):
    """
    Plot single matrix: coherences, coherence differences, phases, PCs or ICs
    (specified in data_type: 'Coherence', 'Coherence diff', 'Phase', 'PCA' or 'ICA') 

    Colors of plots will vary depending on dataType
    """
    if kwargs and ('cmap' in kwargs):
        cmap = kwargs['cmap']
        del kwargs['cmap']
    else:
        cmap = matrix_cmap(dataType)

    if kwargs and ('clim' in kwargs):
        clim = kwargs['clim']
        del kwargs['clim']
    else:
        Vm = np.percentile(data, 99)
        if dataType == 'Coherence':
            clim = [0, Vm]
        elif (dataType == 'Phase') or (dataType == 'PCA') or (dataType == 'ICA'):
            clim = [-Vm, Vm]
        else:
            clim = [-Vm, Vm]

    if ax is not None and fig is not None:
        im = ax.imshow(data, clim=clim, cmap=cmap,
                       interpolation='none', aspect='auto', **kwargs)
        fig.colorbar(im, ax=ax)
    else:
        plt.imshow(data, clim=clim, cmap=cmap,
                   interpolation='none', aspect='auto', **kwargs)
        plt.colorbar()


def matrix_cmap(dataType):

    if dataType == 'Coherence':
        cmap = 'Blues'
    elif (dataType == 'Phase') or (dataType == 'PCA') or (dataType == 'ICA'):
        cmap = 'RdGy'
    else:
        cmap = 'Greys'

    return cmap
