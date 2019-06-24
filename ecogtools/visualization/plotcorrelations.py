"""
Correlation plots, eg scatter, linear regression etc

2015-07-28, LKirkby
"""

import numpy as np
import matplotlib.pylab as plt
import seaborn.apionly as sns
from scipy.stats import linregress


def scatter_linregress(x, y, ax='n/a', zeroX=False, **kwargs):
    
    x = np.array(x)
    y = np.array(y)
        
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):

        goodInds = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        
        x = x[goodInds]
        y = y[goodInds]
    
    if ax == 'n/a':
        fig, ax = plt.subplots(figsize=(9,8))        
    
    xran = max(x)-min(x)
    yran = max(y)-min(y)

    slope, intercept, r, p, stderr = linregress(x, y)
        
    xfit = np.linspace(min(x), max(x), 100)
    yfit = slope*xfit + intercept    
    
    if not kwargs:
        ax.scatter(x, y, c='k', s=30, marker='o')
    else:
        ax.scatter(x, y, **kwargs)
    ax.plot(xfit, yfit, 'k-', linewidth=0.5)
    ax.text(0.95, 0.95, 'r-squared: '+str(round(r**2,2)), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
    if round(p,2) < 0.01:
        ax.text(0.95, 0.9, 'p-value: <0.01 (%s)' % float('%.1g' % p), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)                
    else:
        ax.text(0.95, 0.9, 'p-value: '+str(round(p,2)), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)

    ax.set_ylim((min(y)-(0.1*yran), max(y)+(0.1*yran)))
    if zeroX == True:
        ax.set_xlim((0, max(x)+(0.1*xran)))
    else:
        ax.set_xlim((min(x)-(0.1*xran), max(x)+(0.1*xran)))

    return slope, intercept, r, p, stderr



def scatter_regplot(x, y, ax='n/a', zeroX=False, **kwargs):

    x = np.array(x)
    y = np.array(y)
        
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):

        goodInds = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
        
        x = x[goodInds]
        y = y[goodInds]
        
    if ax == 'n/a':
        fig, ax = plt.subplots(figsize=(9,8))
    
    xran = max(x)-min(x)
    yran = max(y)-min(y)

    slope, intercept, r, p, stderr = linregress(x, y)   
    
    if not kwargs:
        sns.regplot(x, y, ax=ax, scatter_kws={'s': 40}, color='k', marker='o')
    else:
        sns.regplot(x, y, ax=ax, **kwargs)
    ax.text(0.95, 0.95, 'r-squared: '+str(round(r**2,2)), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
    if round(p,2) < 0.01:
        ax.text(0.95, 0.9, 'p-value: <0.01 (%s)' % float('%.1g' % p), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)                
    else:
        ax.text(0.95, 0.9, 'p-value: '+str(round(p,2)), horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)
    
    ax.set_ylim((min(y)-(0.1*yran), max(y)+(0.1*yran)))
    if zeroX == True:
        ax.set_xlim((0, max(x)+(0.1*xran)))
    else:
        ax.set_xlim((min(x)-(0.1*xran), max(x)+(0.1*xran)))
