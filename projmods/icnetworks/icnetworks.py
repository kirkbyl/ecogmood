# @kirkbyl, 30 June 2016
"""
Class for data analysis of Intrinsic Connectivity Networks:
Extract and save ICNs, plot ICNs

21 Nov 2016
Added analysis class for plotting ICNs graphically:
    - plotting intrinsic coherence networks (ICNs) as chord diagram (circular graphical representation)
    - finding common ICNs across patients
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.cm as cm
import itertools
import collections

from ecogtools.analysis import coherences as cohmodule
from ecogtools.tools import loaddata, utilities as utils
from ecogtools.visualization import plotcoherence, plotchord
from ecogtools.recordingparams import subjects as subj


class IntrinsicNetwork(object):
    
    def __init__(self, pID, band, dataPath=None, commonRegbins='default_subset', method='ICA'):
        
        if isinstance(pID, str):
            self.patient = subj.Patient(pID, dataPath=dataPath)
            self.patientID = pID
        elif isinstance(pID, subj.Patient):
            self.patient = pID
            self.patientID = self.patient.ID
        self.band = band
        
        self.allBands = ['theta', 'alpha', 'beta', 'gamma']      
                
        # Template for common regions defined in common_reg_bins function
        self.regBins = common_reg_bins(regbins=commonRegbins)
        
        self.method = method
        if method.upper() == 'ICA':
            dirID = 'ICN'
        elif method.upper() == 'NMF':
            dirID = 'NMF'

        ## NOTE if self.band is 'all' some attributes are not meaningful, eg self.coh is not loaded if band == 'all'
        if band == 'all':
            for bb, bnd in enumerate(self.allBands):
                
                icnDF = loaddata.icn_dataframe(self.patient.ID, self.patient.ICNpath)
                icnOut = loaddata.reload_icn(icnDF, bnd)                
                
                if bb == 0:
                    self.ICN = icnOut['ICN']
                    self.proj = icnOut['proj']
                else:
                    self.ICN = np.dstack((self.ICN, icnOut['ICN']))
                    self.proj = np.vstack((self.proj, icnOut['proj']))
            self.coh = None
            self.cohAve = None
            
        else:
            icnDF = loaddata.icn_dataframe(self.patient.ID, self.patient.ICNpath)
            icnOut = loaddata.reload_icn(icnDF, self.band)
            self.ICN = icnOut['ICN']
            self.proj = icnOut['proj']
            self.coh = icnOut['coh']
            self.cohAve = icnOut['cohAve']
                    
        self.tAxis = icnOut['tAxis']           
        self.regs = icnOut['regs']                
                                                
        return
    
    def printRegions(self):
        
        for kk in self.regs.keys():
            print('{0}: {1}'.format(kk, self.regs[kk]))
        
        return

        
    def rerunICN(self, **kwargs):
                
        # Find ICNs and projections
        icnMat = cohmodule.find_ICs_from_PCA(self.coh, **kwargs)    
        icnProj = cohmodule.project_onto_component(self.coh, icnMat, symmMat=True)
        
        self.newICN = {'ICN': icnMat, 'proj': icnProj}
        
        return self.newICN
        
        
    def saveNewICN(self):
        
        if not hasattr(self, 'newICN'):
            print('No new ICN to save')
        
        saveDir = self.patient.ICNpath
        folderIDs = self.newICN.keys()
        
        for folderID in folderIDs:
            saveSubdir = saveDir+'/'+folderID
            filename = saveSubdir+'/'+self.patientID+'_'+folderID+'_'+self.band
            np.save(filename, self.newICN[folderID])
        
        return
        
        
    def runPCA(self, nPCs=50, matType='half', meanSubtract=True, plotPCA=True, subset=range(10), saveFig=False, figname='PCA', saveDir='default'):
        
        PCs, eigVs = cohmodule.pca_calculation(self.coh, nPCs=nPCs, pcaType=matType, meanSubtract=meanSubtract)
        
        if plotPCA == True:
            fig = plotcoherence.plot_matrices(PCs, dataType='PCA', subset=subset)
            
            if saveFig == True:
                self.saveFig(fig, figname=figname, saveDir=saveDir)        
            return
                        
                        
    def getAveICNs(self):
        
        self.aveICN = cohmodule.average_coh_matrices(self.ICN, self.regs)
        
        return
        

    def getCommonICNs(self, regBins=None, nBinSplit=4):
        """
        Construct 'common' ICNs using average ICN and pre-defined regions that are most common across patients
        
        For my analysis, common regions are defined by commmon_reg_bins function
        *** refer to ecogtools.recordingparams.elecs for abbreviation definitions ***)
        
        Then take first 4 electrodes (nBinSplit) -- ie four most medial electrodes, whether 4, 6, or 10 electrodes on a given lead
        """
        
        if regBins is None:
            regBins = self.regBins

                                        
        self.commonICN = np.zeros((len(regBins)*nBinSplit, len(regBins)*nBinSplit, self.ICN.shape[-1]))
        
        regPairs = utils.get_pairs(regBins, regBins, includeSame=True, unique=False)
        
        regMap = self.getCommonRegmap(regBins=regBins, nBinSplit=nBinSplit)

        regdone1 = ''
        regdone2 = ''
        
        regpass1 = ''
        regpass2 = ''
        
        # Keep track of which common regions do not have electrodes
        self.emptyNodes = []        
        
        for rp, regPair in enumerate(regPairs):
            
            reg1 = regPair[0]
            reg2 = regPair[1]
            
            inds1 = regMap[reg1]
            inds2 = regMap[reg2]
                    
            if reg1 in self.regs:
                elecs1 = self.regs[reg1]
            elif reg1[1:] in self.regs:
                
                if reg1[1:] not in regdone1 or reg1 in regpass1:
                    elecs1 = self.regs[reg1[1:]]
                    regdone1 = reg1[1:]
                    regpass1 = reg1
                else:
                    elecs1 = None
                    if inds1[0] not in self.emptyNodes:
                        self.emptyNodes.extend(inds1)
                                       
            else:
                elecs1 = None            
                if inds1[0] not in self.emptyNodes:
                    self.emptyNodes.extend(inds1)       
                     
            if reg2 in self.regs:
                elecs2 = self.regs[reg2]
            elif reg2[1:] in self.regs:
                
                if reg2[1:] not in regdone2 or reg2 in regpass2:
                    elecs2 = self.regs[reg2[1:]]
                    regdone2 = reg2[1:]
                    regpass2 = reg2
                else:
                    elecs2 = None
                
            else:
                elecs2 = None                              
                  
            if elecs1 is not None and elecs2 is not None:

                for ii, ee in zip(inds1, elecs1[0:nBinSplit]):
                    self.commonICN[ii, inds2, :] = self.ICN[ee, elecs2[0:nBinSplit], :]
            
                            
        return
    
    def getCommonRegmap(self, regBins=None, nBinSplit=4):
         
        if regBins is None:
            regBins = self.regBins
            
        self.regMap = collections.OrderedDict()
        for rr, regBin in enumerate(regBins):
            rStart = rr*nBinSplit
            
            self.regMap[regBin] = range(rStart, rStart+nBinSplit)
         
        return self.regMap                           
        
    def plotICN(self, subset='default', saveFig=False, figname='ICN', saveDir='default', plotNew=False, plotAve=True, plotCommon=True, **kwargs):
        
        if plotNew == True:
            if not hasattr(self, 'newICN'):
                print('Running new ICN')
                self.newICN = self.rerunICN()
            plotICN = self.newICN['ICN']
        elif plotNew == False:
            plotICN = self.ICN
            
        if subset == 'default':
            subset = range(np.atleast_3d(plotICN.shape[-1])[-1])
        else:
            subset = subset
        
        title = '{0}, {1} band'.format(self.patientID, self.band)
        fig = plotcoherence.plot_matrices(plotICN, dataType='ICA', subset=subset, title=title, **kwargs)
        if plotAve == True:
            if not hasattr(self, 'aveICN'):
                self.getAveICNs()
            plotcoherence.plot_matrices(self.aveICN, dataType='ICA', subset=subset, title=title+', ave ICNs', **kwargs)
        if plotCommon == True:
            if not hasattr(self, 'commonICN'):
                self.getCommonICNs()
            plotcoherence.plot_matrices(self.commonICN, dataType='ICA', subset=subset, title=title+', common ICNs', **kwargs)
            
        if saveFig == True:
            self.saveFig(fig, figname=figname, saveDir=saveDir)        
        return
        
        
    def plotCoh(self, subset='default', saveFig=False, figname='Coh', saveDir='default', **kwargs):
        
        if subset == 'default':
            subset = range(10)
        else:
            subset = subset
       
        fig = plotcoherence.plot_matrices(self.coh, dataType='Coherence', subset=subset)
        
        if saveFig == True:
            self.saveFig(fig, figname=figname, saveDir=saveDir)        
        return
        
        
    def saveFig(self, fig, figname='default', saveDir='default', figtype='eps', **kwargs):
                
        if figname == 'default':
            figname = self.patientID+'_'+self.band
        else:
            figname = self.patientID+'_'+self.band+'_'+figname
            
        utils.saveFig(fig, figname=figname, saveDir=saveDir, figtype=figtype, **kwargs)
        
        return


 
               
class IcnGraph(object):
    """
    Graphical representation of intrinsic connectivity networks
    """
            
    def __init__(self, pID, band, dataPath=None, method='ICA', graphType='common', quantileThresh='default', weightThresh='default'):
        """
        - pID: patient ID, string or Patient class
        - band: frequency band of corresponding Intrinsic Network
        - dataPath: path to patient ICN data
        - quantileThresh: threshold for constructing graph edges from upper quantile of ICN loading (ie if quantileThresh = 98, ICN loading > 98th percentile will count as edge)
        - weightThresh: threshold for constructing graph edges from relative weight of ICN loading (eg if weightThresh = 0.3, normalized ICN loading > 0.3 will count as edge)
        
        - graphType can be 'full', 'ave', or 'common':
            - full: all electrodes
            - ave: average across electrodes in given brain region
            - common: 'common' network (across patients) defined by these regions: ['aSTC', 'mSTC', 'pSTC', 'AM', 'vHPC', 'mHPC', 'dHPC', 'aOFC', 'pOFC', 'iCIN', 'sCIN', 'dCIN', 'INS']
        """
        
        ICN = IntrinsicNetwork(pID, band, dataPath=dataPath, method=method)
        self.patientID = ICN.patientID
        self.band = band
        self.regs = ICN.regs
        
        # This is hard coded for my analysis -- I have chosen common regions for my ICN analysis (['STC', 'AM', 'HPC', 'OFC', 'CIN', 'INS'], defined in saveICN.py)
        self.regList = ['STC', 'AM', 'HPC', 'OFC', 'CIN', 'INS']
        self.regBins = ICN.regBins
        self.regMap = ICN.getCommonRegmap()
        
        self.graphType = graphType
        
        if graphType == 'full':
            self.graphICN = ICN.ICN
            if quantileThresh == 'default':
                quantileThresh = 98
            else:
                quantileThresh = quantileThresh
                
        elif graphType == 'ave':
            ICN.getAveICNs()
            self.graphICN = ICN.aveICN
            if weightThresh == 'default':
                weightThresh = 0.3
            else:
                weightThresh = weightThresh            
                
                        
        elif graphType == 'common':
            ICN.getCommonICNs()
            self.graphICN = ICN.commonICN
            self.emptyNodes = ICN.emptyNodes

            if weightThresh == 'default':
                weightThresh = 0.3
            else:
                weightThresh = weightThresh      
            if quantileThresh == 'default':
                quantileThresh = 98
            else:
                quantileThresh = quantileThresh
                
        self.constructGraphs(quantileThresh=quantileThresh, weightThresh=weightThresh)
        
        return
        
    def constructGraphs(self, quantileThresh=98, weightThresh=0.3):
        """
        Creates dictionary of graphs correponding to each ICN: {ICNint: {'pos': networkx graph, 'neg': networkx graph}}
        
        Each graphical representation of each ICN is overlay of 'pos' graph (increased connectivity) and 'neg' graph (decreased connectivity)
        
        - ICNint key: integer corresponding to ICN number (indexed from 0)
        - 'pos' graph: edges that correspond to increase in connectivity in given ICN
        - 'neg' graph: edges that correspond to decrease in connectivity in given ICN
        - quantileThresh: threshold for constructing graph edges from upper quantile of ICN loading (ie if quantileThresh = 98, ICN loading > 98th percentile with count as edge)
        """
        
        nodes = self.getNodes()
        if self.graphType == 'full' or self.graphType == 'common':
            edges = self.getEdges(quantile=quantileThresh)
        elif self.graphType == 'ave':
            edges = self.getWeightedEdges(weightThresh=weightThresh)
        
        self.graphs = {}
        
        for kk in range(self.graphICN.shape[-1]):
            
            self.graphs[kk] = {}
            
            for pp, key in enumerate(['pos', 'neg']):
                G = nx.Graph()
                G.add_nodes_from(nodes)
                if key == 'pos':
                    if self.graphType == 'full' or self.graphType == 'common':
                        G.add_edges_from(edges[kk][0])
                    elif self.graphType == 'ave':
                        G.add_weighted_edges_from(edges[kk][0])
                elif key == 'neg':
                    if self.graphType == 'full' or self.graphType == 'common':
                        G.add_edges_from(edges[kk][1])
                    elif self.graphType == 'ave':
                        G.add_weighted_edges_from(edges[kk][1])
                                            
                self.graphs[kk][key] = G
            
        return
            
        
    def plotGraphs(self, subset='default', radius=10, noderadius='default', alpha='default', linewidth='default', plotSeparate=False, fig=None, ax=None, colormap=cm.plasma):
        """
        Plot chord diagram of ICN graphs
        
        - radius: radius for nodes in chord diagram
        - alpha: transparency of edges
        - noderadius: radius of nodes
        - plotSeparte: plot each graph in separate figure if True, plot in subplots if False
        """
                        
        if subset == 'default':
            subset = np.array(self.graphs.keys())
        else:
            subset = subset
        
        nodeLabels = self.getNodeLabels()
        nodecolor = self.getNodeColors(colormap=colormap)
        nodefill = self.getNodeFills()
        
        if noderadius == 'default':
            if self.graphType == 'full' or self.graphType == 'common':
                noderadius = 0.07
            elif self.graphType == 'ave':
                noderadius = 0.2
        else:
            noderadius = noderadius

        if alpha == 'default':
            if self.graphType == 'full' or self.graphType == 'common':
                alpha = 0.3
            elif self.graphType == 'ave':
                alpha = 1
        else:
            alpha = alpha 

        if linewidth == 'default':
            linewidth = 1
        else:
            linewidth = linewidth       
                        
            
        title = 'Patient {0}, {1} band'.format(self.patientID, self.band)
        
        if plotSeparate == False:
            fig = plotchord.plot_graphs(self.graphs, subset=subset, title=title, nodeLabels=nodeLabels, radius=radius, noderadius=noderadius, alpha=alpha, nodecolor=nodecolor, nodefill=nodefill, linewidth=linewidth, fig=fig, ax=ax, colormap=colormap)
        elif plotSeparate == True:
            for ss in subset:
                fig = plotchord.plot_graphs(self.graphs, subset=[ss], title=title, nodeLabels=nodeLabels, radius=radius, noderadius=noderadius, alpha=alpha, nodecolor=nodecolor, nodefill=nodefill, linewidth=linewidth, fig=fig, ax=ax, colormap=colormap)
        plt.show()            
        
        return fig
        
        
    def getNodes(self):
        
        nodes = range(self.graphICN.shape[0])
                           
        return nodes
        
        
    def getNodeLabels(self):
        
        if self.graphType == 'full':
            
            nodeLabels = plotchord.get_node_labels(self.regList, self.regs)
            
        elif self.graphType == 'ave':

            averegs = {}
            for kka, keya in enumerate(self.regs.keys()):
                averegs[keya] = [kka]
                            
            nodeLabels = plotchord.get_node_labels(self.regList, averegs)
            
        elif self.graphType == 'common':
                                
            nodeLabels = plotchord.get_node_labels(self.regList, self.regMap)
        
        return nodeLabels
        
        
    def getNodeColors(self, colormap=cm.plasma):
    
        nodeLabels = self.getNodeLabels()
            
        nodecolor = plotchord.get_node_colors(nodeLabels, colormap=colormap)
        
        return nodecolor
        
        
    def getNodeFills(self):
    
        nodeLabels = self.getNodeLabels()
        
        if hasattr(self, 'emptyNodes'):
            emptyNodes = self.emptyNodes
        else:
            emptyNodes = None
            
        nodefill = plotchord.get_node_fills(nodeLabels, emptyNodes=emptyNodes)
        
        return nodefill    
            
        
    def getEdges(self, quantile=98):
        """
        Get edges for all ICNs (edges correspond to ICN loading greater than or less than 'quantile' threshold)
        
        Returns dictionary of edges:
            - key: ICN number
            - values: 2-element list of edges. First element corresponds to >quantile and second corresponds to <100-quantile
                (ie. connectivity between regions that increase vs. decrease in ICN)
        """
        
        edges = {}
        self.icnBin = np.zeros_like(self.graphICN)
        
        for kk, icn in enumerate(np.rollaxis(self.graphICN, 2)):
                    
            thresh = np.percentile(abs(icn), quantile)
            
            # ICN loading can be + or -. Define threshold based on which abs(percentile) is greatest
            neg = np.percentile(icn, 100-quantile)
            pos = np.percentile(icn, quantile)
                
            if abs(neg)>pos:
                icnSigned = -icn
            else:
                icnSigned = icn 
                
            edges[kk] = []
                
            for tt in (thresh, -thresh):
                
                # Connections with increased connectivity
                if tt > 0:
                    ee = [ii for ii in zip(np.where(icnSigned>tt)[0], np.where(icnSigned>tt)[1])]
                    
                    for mm, nn in zip(np.where(icnSigned>tt)[0], np.where(icnSigned>tt)[1]):
                        self.icnBin[mm, nn, kk] = 1
                   
                # Connections with decreased connectivity
                elif tt < 0:
                    ee = [ii for ii in zip(np.where(icnSigned<tt)[0], np.where(icnSigned<tt)[1])]   
                    
                    for mm, nn in zip(np.where(icnSigned<tt)[0], np.where(icnSigned<tt)[1]):
                        self.icnBin[mm, nn, kk] = -1
                                    
                edges[kk].append(ee)                    
                                        
        return edges
        
    def plotBinerized(self, quantile=98, subset='default'):
        
        if not hasattr(self, 'icnBin'):
            self.getEdges(quantile=quantile)
            
        if subset == 'default':
            subset = range(np.atleast_3d(self.icnBin.shape[-1])[-1])
        else:
            subset = subset
                    
        title = 'Binerized ICNs for patient {0}, {1} band'.format(self.patientID, self.band)
        plotcoherence.plot_matrices(self.icnBin, subset=subset, dataType='ICA', title=title)

        
    def getWeightedEdges(self, weightThresh=0.3):
        
        edges = {}
        
        nodes = self.getNodes()
        
        edgePairs = utils.get_pairs(range(len(nodes)), range(len(nodes)), includeSame=False, unique=True)
        
        for kk, icn in enumerate(np.rollaxis(self.graphICN, 2)):
            
            edges[kk] = []
            edges[kk].append([])
            edges[kk].append([])
            
            
            icnNorm = np.copy(icn)
            if icn.max() >= abs(icn.min()):
                normFac = icn.max()
            else:
                normFac = icn.min()
            
            icnNorm = icn/normFac
            
            for pair in edgePairs:
                weight = icnNorm[pair[0], pair[1]]
                
                if weight > weightThresh:
                    edges[kk][0].append((pair[0], pair[1], weight))
                
                elif weight < -weightThresh:
                    edges[kk][1].append((pair[0], pair[1], weight))
                    
        return edges


    def getEdgeColors(self, edges):
    
        nodeLabels = self.getNodeLabels()
        edgecolor = plotchord.get_edge_colors(edges, nodeLabels)
        
        return edgecolor                    
          


def common_reg_bins(regbins='default_all'):
    """
    regbins can be:
        - default_all
        - default_subset (removes mHPC, dHPC, dCIN, which are less common electrode locs in patients)
        - list of desired reg bins
    """
    
    regbinlist = []
    if regbins == 'default_all':
        regbinlist = ['aSTC', 'mSTC', 'pSTC', 'AM', 'vHPC', 'mHPC', 'dHPC', 'aOFC', 'pOFC', 'iCIN', 'sCIN', 'dCIN', 'INS']
    elif regbins == 'default_subset':
        regbinlist = ['aSTC', 'mSTC', 'pSTC', 'AM', 'vHPC', 'aOFC', 'pOFC', 'iCIN', 'sCIN', 'INS']
    else:
        regbinlist = regbins

    return regbinlist
