# 14 June 2016, @kirkbyl

import matplotlib.dates as mdates
from scipy.spatial import distance
import time
import datetime
import matplotlib.pylab as plt
import numpy as np
import re
import os
import matplotlib
matplotlib.use('Agg')


def find_nearest(array, value):
    """
    Find element and index of array nearest to value

    Array and value can also be n-dimensional vector
    """

    if np.squeeze(np.array(array)).shape == 1:

        # 1-dimensional
        idx = (np.abs(array-value)).argmin()

    else:
        # n-dimensional
        idx = distance.cdist(np.array(array), np.atleast_2d(value)).argmin()

    return idx, array[idx]


def str_to_list(strtoseparate):
    """
    Turns comma separated strings into a list
    """

    return [aa.strip() for aa in strtoseparate.split(',')]


def get_pairs(list1, list2, includeSame=False, unique=True):
    """
    Get all pairs (unique or not) between elements in list1 and list2.

    If same element is in both lists include pair if includeSame == True.
    """

    pairs = []
    for aa in list1:
        for bb in list2:
            if aa == bb and includeSame == False:
                pass
            else:
                if unique == True:
                    if (bb, aa) not in pairs:
                        pairs.append((aa, bb))
                elif unique == False:
                    pairs.append((aa, bb))

    return pairs


def custom_subplots(nSubplots, colorbar=False, figsize=None, **kwargs):
    """
    Standard figure for creating n subplots
    """

    sz = int(np.ceil(np.sqrt(nSubplots)))

    if (sz**2-sz) >= nSubplots:
        nRow = sz-1
        nCol = sz
    else:
        nRow = sz
        nCol = sz

    if figsize is None:
        if nSubplots == 1:
            if colorbar == True:
                figsize = (8, 6)
            elif colorbar == False:
                figsize = (7, 7)
        elif nSubplots == 2:
            if colorbar == True:
                figsize = (14, 5)
            elif colorbar == False:
                figsize = (12, 5)
        else:
            if nRow == nCol:
                if colorbar == True:
                    figsize = (12, 9)
                elif colorbar == False:
                    figsize = (10, 9)
            elif nRow != nCol:
                if colorbar == True:
                    figsize = (12, 6)
                elif colorbar == False:
                    figsize = (10, 6)

    fig, ax = plt.subplots(nRow, nCol, figsize=figsize, **kwargs)
    subplotPairs = get_pairs(range(nRow), range(
        nCol), includeSame=True, unique=False)

    for ss in subplotPairs[nSubplots:]:
        ax[ss[0], ss[1]].axis('off')

    ax1D = np.ravel(ax)

    return fig, ax, ax1D


def is_power(n, base):
    """
    Check if n is a power of base
    """

    if n == base:
        return True

    if base == 1:
        return False

    temp = base

    while (temp <= n):
        if temp == n:
            return True
        temp *= base

    return False


def rand_jitter(var, stdev=0.01):
    """
    Create random jitter on input array or integer
    """
    if isinstance(var, int):
        varJitter = var+np.random.randn(1)*stdev

    else:
        if max(var)-min(var) == 0:
            stdev = stdev
        else:
            stdev = stdev*(max(var)-min(var))

        varJitter = var+np.random.randn(len(var))*stdev

    return varJitter


def tryint(ss):
    try:
        return int(ss)
    except:
        return ss


def alphanum_key(ss):
    """ 
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(cc) for cc in re.split('([0-9]+)', ss)]


def sort_natural(inputList):
    """ 
    Sort list in the way that humans expect (1, 2, 3, ... 10, 11 etc)
    """
    inputList.sort(key=alphanum_key)

    return inputList


def print_full_dataframe(x):
    """
    Display all rows in dataframe
    """

    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

    return


def inv_dict(dictionary):
    """
    Invert dictionary keys and values
    """

    inv_dict = dict(zip(dictionary.values(), dictionary.keys()))

    return inv_dict


def convert_timestamp(tStamp, isTuple=False):
    """
    Converts timestamp to UTC, datetime or matplotlib time

    Inputs:
        - tStamp: can be list or single timestamp. Input may be UTC or datetime object 
        - convertTo: choose from 'datetime', 'UTC', 'matplotlib' or 'all' (returns all three)
        - printOutput: True if want display conversion
    """

    tUTC = []
    tDatetime = []
    tMatplotlib = []

    if isinstance(tStamp, list) == False:
        tList = [tStamp]
        if isTuple == False:
            try:
                tList = list(tStamp)
            except:
                tList = [tStamp]
        elif isTuple == True:
            tList = [tStamp]
    elif isinstance(tStamp, list) == True:
        tList = tStamp

    t0 = tList[0]
    if isinstance(t0, int) or isinstance(t0, float):
        tUTC = tList
        tDatetime = [datetime.datetime.fromtimestamp(tt) for tt in tList]
        tMatplotlib = [mdates.date2num(dt) for dt in tDatetime]

    elif isinstance(t0, datetime.datetime):
        tDatetime = tList
        tMatplotlib = [mdates.date2num(dt) for dt in tList]
        tUTC = [int(time.mktime(tt.timetuple())) for tt in tList]

    elif isinstance(t0, tuple):
        tDatetime = [datetime.datetime(
            tt[0], tt[1], tt[2], tt[3], tt[4], tt[5]) for tt in tList]
        tUTC = [int(time.mktime(dt.timetuple())) for dt in tDatetime]
        tMatplotlib = [mdates.date2num(dt) for dt in tDatetime]

    if len(tList) == 1:
        tUTC = tUTC[0]
        tDatetime = tDatetime[0]
        tMatplotlib = tMatplotlib[0]

    return {'UTC': tUTC, 'datetime': tDatetime, 'matplotlib': tMatplotlib}
