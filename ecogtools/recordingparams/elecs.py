# 2015-08-31, LKirkby

import collections
import pandas as pd

from ecogtools.tools import utilities as utils

def electrode_locs(patientID):
    """
    Electrode mapping
    
    Subdural grid: subdural grid (lateral frontal temporal cortex): lFTC_G
    inf Frontal: inferior frontal cortex: iFC
    lat Temporal: lateral temporal: lTC
    ant ST: anterior subtemporal: aSTC
    mid ST: middle subtemporal: mSTC
    pos ST: posterior subtemporal: pSTC
    OFC: orbito-frontal cortex: OFC
    ant OFC: anterior orbito-frontal cortex: aOFC
    pos OFC: posterior orbito-frontal cortex: pOFC
    Amygdala: amygdala depth: AM
    Hippocampus: hippocampus depth: HPC
    inf CIN: inferior anterior cingulate: iACC
    sup CIN: superior anterior cingulate: sACC
    pos CIN: posterior cinculate: PCC
    INS: insular: INS
    ant INS: anterior insular: aINS
    pos INS: posterior insular: pINS
    temporal pole/frontal pole: TP/FP
            
    STC: anterior or posterior (a, p)
    CIN: 2 axes: inferior/superior (in anterior part of CIN - i, s) or anterior/posterior (pCIN)
    INS: anterior or posterior (a, p)
    OFC: anterior or posterior (a, p)
    HPC: ventral/anterior or dorsal/posterior (v, d)
    
    Mesial depth: anterior MES is often AM and posterior MES is often HPC
    
    Bilateral electrodes: _R or _L
    Grid: _G
    Depths, strips: no denotation
    """
    
    # Tuples in regions_nElecs are defined by ('Name', 'Abbr', nElecs)

    if patientID == 'EC77':
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('ITG', 'ITG', 6), ('pos Temporal', 'pTC', 6), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('INS', 'INS', 4), ('sup CIN', 'sCIN', 4), ('inf CIN', 'iCIN', 4), ('Amygdala', 'AM', 4), ('Hippocampus', 'HPC', 4)]                        
        nElecsTot = 116
        
    elif (patientID == 'EC79'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('right Temporal', 'rTC', 4), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('Amygdala', 'AM', 4), ('Hippocampus', 'HPC', 4), ('sup CIN', 'sCIN', 4), ('inf CIN', 'iCIN', 4), ('INS', 'INS', 4)]                       
        nElecsTot = 108
        
    elif (patientID == 'EC80') :
        regions_nElecs = [('Amygdala', 'AM', 10), ('ER-ctx', 'ERC', 10), ('Hipp-head', 'vHPC', 10), ('Hipp-body', 'mHPC', 10), ('Hipp-tail', 'dHPC', 10), ('Cuneus', 'CUN', 10), ('Calcarine', 'CAL', 10), ('inf CIN', 'iCIN', 10), ('INS', 'INS', 10), ('sup CIN', 'sCIN', 10), ('OFC', 'OFC', 10), ('Lingual-gyrus', 'LGY', 10)]                       
        nElecsTot = 120
                                                      
    elif (patientID == 'EC82') or (patientID == 'EC82a') or (patientID == 'EC82b'):
        # renamed superior OFC -> ant OFC; inferior OFC -> pos OFC for consistency with other patients
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('ant Frontal', 'aFC', 20), ('ant Hipp', 'AM', 4), ('pos Hipp', 'HPC', 4), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ITG', 'ITG', 6), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('mid Frontal', 'mFC', 4)]                       
        nElecsTot = 122
        
    elif patientID == 'EC84':
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('ant Temporal', 'aTC', 20), ('ITG', 'ITG', 6), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('sup CIN', 'sCIN', 4), ('inf CIN', 'iCIN', 4), ('INS', 'INS', 4), ('ant MES', 'AM', 4), ('pos MES', 'HPC', 4)]                       
        nElecsTot = 126
        
    elif (patientID == 'EC108'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('OFC grid', 'OFC_G', 32), ('dl frontal grid', 'dlFC_G', 32), ('temporal pole', 'TP', 6), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ITG', 'ITG', 6), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('CIN', 'CIN', 10)]
        nElecsTot = 188             
              
    elif (patientID == 'EC113'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('inf temp gyrus', 'iTGY', 6), ('Amygdala', 'AM', 4), ('Hippocampus', 'HPC', 4), ('ant ST',  'aSTC', 6), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('INS', 'INS', 10), ('OFC grid', 'OFC_G', 64), ('lat temp grid', 'lTC_G', 20)]
        nElecsTot = 186
        
    elif (patientID == 'EC122'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('Heschls', 'HES', 10), ('OFC', 'OFC', 4)]
        nElecsTot = 122
        
    elif (patientID == 'EC125'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('INS', 'INS', 10), ('Amygdala', 'AM', 10), ('sup CIN', 'sCIN', 10), ('Hippocampus', 'HPC', 10), ('inf CIN', 'iCIN', 10), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('temporal pole', 'TP', 6), ('inf temp gyrus', 'iTGY', 6)]
        nElecsTot = 146
        
    elif (patientID == 'EC129'):
        regions_nElecs = [('Subdural grid1', 'lFC_G', 32), ('Subdural grid2', 'lTC_G', 32), ('inf frontal pole', 'iFP', 6), ('sup frontal pole', 'sFP', 6), ('ant frontal interhemipsheric', 'aFIH', 6), ('pos frontal interhemipsheric', 'pFIH', 6), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 4), ('Hippocampus', 'HPC', 4), ('inf CIN', 'iCIN', 4), ('sup CIN', 'sCIN', 4), ('INS', 'INS', 10), ('parietal A', 'par A', 6), ('parietal B', 'par B', 6), ('sup frontal', 'sFC', 6)]
        nElecsTot = 154

    elif (patientID == 'EC131'):
        regions_nElecs = [('Frontal grid', 'FC_G', 32), ('Temporal grid', 'TC_G', 32), ('temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('inf CIN', 'iCIN', 10), ('sup CIN', 'sCIN', 10)]
        nElecsTot = 136

    elif (patientID == 'EC133'):
        regions_nElecs = [('Temporal grid', 'TC_G', 32), ('Frontal grid', 'FC_G', 32), ('ant temporal pole', 'aTP', 6), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ITG', 'ITG', 6), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('CIN', 'CIN', 10), ('INS', 'INS', 10), ('OFC grid', 'OFC_G', 64)]
        nElecsTot = 192    
      
    elif (patientID == 'EC136'):
        regions_nElecs = [('Temporal grid', 'TC_G', 32), ('Frontal grid', 'FC_G', 32), ('OFC grid', 'OFC_G', 64), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('temporal pole', 'TP', 4), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('inf lat frontal', 'ilFC', 6), ('sup lat frontal', 'slFC', 6), ('inf CIN', 'iCIN', 10), ('sup CIN', 'sCIN', 10), ('INS', 'INS', 10)]
        nElecsTot = 202
              
    elif (patientID == 'EC139'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('lat temporal A', 'lTC A', 6), ('lat temporal B', 'lTC B', 6), ('INS', 'INS', 10), ('sup CIN', 'sCIN', 10), ('inf CIN', 'iCIN', 10), ('OFC', 'OFC', 4), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4)]
        nElecsTot = 138
        
    elif (patientID == 'EC142'):
        regions_nElecs = [('Frontal grid', 'FC_G', 32), ('Temporal grid', 'TC_G', 32), ('Frontal pole', 'FP', 6), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('Temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4), ('pos ST', 'pSTC', 4), ('ITG', 'ITG', 6), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('ven CIN', 'iCIN', 10), ('dor CIN', 'sCIN', 10), ('INS', 'INS', 10)]
        nElecsTot = 152
        
    elif (patientID == 'EC143'):
        regions_nElecs = [('Intrasylvian grid', 'IS_G', 32), ('Temporal grid', 'TC_G', 32), ('Frontal grid', 'FC_G', 32), ('INS', 'INS', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('OFC', 'OFC', 4), ('Temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('mid ST', 'mSTC', 4)]
        nElecsTot = 138
        
    elif (patientID == 'EC148'):
        regions_nElecs = [('Subdural grid', 'lFTC_G', 64), ('sup Frontal', 'sFC', 4), ('mid Frontal', 'mFC', 6), ('inf Frontal', 'iFC', 6), ('Frontal pole', 'FP', 6), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('ven CIN', 'iCIN', 10), ('dor CIN', 'sCIN', 10)]
        nElecsTot = 152
        
    elif (patientID == 'EC153'):
        regions_nElecs = [('Frontal grid', 'FC_G', 32), ('Temporal grid', 'TC_G', 32), ('ant OFC', 'aOFC', 4), ('pos OFC', 'pOFC', 4), ('temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('ven CIN', 'iCIN', 10), ('dor CIN', 'sCIN', 10)]
        nElecsTot = 136

    elif (patientID == 'EC155'):
        regions_nElecs = [('Frontal grid', 'FC_G', 32), ('Temporal grid', 'TC_G', 32), ('ITG', 'ITG', 6), ('OFC', 'OFC', 4), ('temporal pole', 'TP', 6), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('INS', 'INS', 10), ('CIN', 'CIN', 8)]
        nElecsTot = 126
        
    elif (patientID == 'EC156'):
        regions_nElecs = [('Frontal grid', 'FC_G', 32), ('Temporal grid', 'TC_G', 32), ('OFC', 'OFC', 4), ('temporal pole', 'TP', 6), ('inf temp gyrus', 'iTGY', 4), ('ant ST', 'aSTC', 4), ('pos ST', 'pSTC', 4), ('Hippocampus', 'HPC', 10), ('Amygdala', 'AM', 10), ('INS', 'INS', 10), ('ven CIN', 'iCIN', 10), ('dor CIN', 'sCIN', 10)]
        nElecsTot = 136
        
    elif (patientID == 'EC158'):
        regions_nElecs = [('sup frontal A', 'sFC_A', 10), ('sup frontal B', 'sFC_B', 10), ('sup frontal C', 'sFC_C', 10), ('CIN', 'CIN', 10), ('Amygdala', 'AM', 10), ('Hippocampus', 'HPC', 10), ('lat frontal A', 'lFC_A', 4), ('lat frontal B', 'lFC_B', 4), ('OFC', 'OFC', 10)]
        nElecsTot = 78

        
    # Convert to dictionary format:
    brainRegions = collections.OrderedDict()
    brainRegions_abbr = collections.OrderedDict()
    brainRegions_map = collections.OrderedDict()
    
    nElecs = 0
    for region in regions_nElecs:
        brainRegions[region[0]] = range(nElecs, nElecs+region[-1])
        brainRegions_abbr[region[1]] = range(nElecs, nElecs+region[-1])
        brainRegions_map[region[1]] = region[0]
        nElecs = nElecs + region[-1]
        
    if nElecs != nElecsTot:
        print('Warning: electrode count mismatch for patient '+patientID)
        nElectrodes = 'mismatch'
    else:
        nElectrodes = nElecs
        
    return {'brainRegions': brainRegions, 'nElectrodes': nElectrodes, 'brainRegions_abbr': brainRegions_abbr, 'brainRegions_map': brainRegions_map}
        
       
        
        
def electrode_locs_dataframe(patientID):
    
    brainRegions = electrode_locs(patientID)['brainRegions']
    brainRegions_map = electrode_locs(patientID)['brainRegions_map']
    
    dfRow = []
    
    for abbr, region in brainRegions_map.items():
        
        electrodes = brainRegions[region]
        dfRow.append({'region': region, 'abbr': abbr, 'electrodes': electrodes})

    cols = ['region', 'abbr', 'electrodes']    
    electrodeInfo = pd.DataFrame(dfRow, columns=cols) 
            
    return electrodeInfo
    
                                                                             
                                                     
def fq_bands():
    
    # Define frequency oscillation bands, range in Hz:
    frequencyBands = collections.OrderedDict()
    frequencyBands['delta'] = (1, 4)
    frequencyBands['theta'] = (4, 8)
    frequencyBands['alpha'] = (8, 13)
    frequencyBands['beta'] = (13, 30)
    frequencyBands['gamma'] = (30, 70)
    frequencyBands['gammaHi'] = (70, 100)

    return frequencyBands   
    
    
    
def choose_electrodes(patientID, abbrs, no64Grid=False, excludeGrids=False, abbrKey=True):
    """
    Get electrode numbers corresponding to brain regions given by abbrs for given patient
    
    Inputs:
        - patientID: string of patient ID eg. 'EC77'
        - abbrs: list or string (comma separated) of abbreviations of brain regions (see electrode_locs for more info)
          eg. abbrs = ['AM', 'HPC', 'OFC']  or 'AM, HPC, OFC', to choose amygdala, hippocampus and OFC electrodes
        - no46Grid: boolean. Some patients eg. EC84 have 64-electrode frontal-parietal grid removed when storing coherence matrices (due to space/memory limitations).
          coherences.is_grid_removed tests whether this grid is removed or not from analysis: boolean output no64Grid
        - excludeGrids: boolean. If True then do not include electrodes located on grid (as opposed to strips/depths)
        - abbrKey: return dictionary with abbreviated electrode labels if True, full electrode labesl if False
        
    Returns:
        - regs: dictionary of {abbr: [electrode numbers]}
        - regsNew: " re-indexed from zero
    """
    
    brainRegions_abbr = electrode_locs(patientID)['brainRegions_abbr']
    brainRegions_map = electrode_locs(patientID)['brainRegions_map']
    
    brainRegions_map_inverted = dict((v, k) for k, v in brainRegions_map.items())
    
    regionKeys = brainRegions_abbr.keys()
    
    regs_temp = collections.OrderedDict()
    regs = collections.OrderedDict()
    
    if isinstance(abbrs, str):
        abbrs = utils.str_to_list(abbrs)
        
    abbrList = []
    for abbr in abbrs:
        if abbr in brainRegions_map_inverted.keys():
            abbrList.append(brainRegions_map_inverted[abbr])
        else:
            abbrList.append(abbr)
    
    for aa in abbrList:
        subset = [rr for rr in regionKeys if aa in rr]
        for ss in subset:
            if no64Grid == True:
                regs_temp[ss] = [bb-64 for bb in brainRegions_abbr[ss]]
            else:
                regs_temp[ss] = brainRegions_abbr[ss]

    if excludeGrids == True:
                
        for key in regs_temp.keys():
            if '_G' in key:
                regs_temp.pop(key)
                
    if abbrKey == True:
        regs = regs_temp
    else:
        for key in regs_temp.keys():            
            regs[brainRegions_map[key]] = regs_temp[key]
                
    regsNew = reindex_brainregions(regs)
        
    return regs, regsNew
    



def reindex_brainregions(regs):
    """
    Reindex brain regions as defined from choose_electrodes
    """
    
    # Reindex regions:
    regsNew = collections.OrderedDict()
    
    nn = 0
    for region in regs.keys():
        elecs = regs[region]
        mm = len(elecs)
        regsNew[region] = range(nn, nn+mm)
        nn = nn + mm    
    
    return regsNew



def get_region_index(patientID, abbrs):
    """
    abbrs: list or string (comma separated) of abbreviations of brain regions (see electrode_locs for more info)
    eg. abbrs = ['AM', 'HPC', 'OFC']  or 'AM, HPC, OFC', to choose amygdala, hippocampus and OFC electrodes
    
    Returns index corresponding to region when averaging across electrodes. Eg subdural grid index is often 0 (first listed region)
    """
    
    electrodeDF = electrode_locs_dataframe(patientID)
    brainRegions_abbr = electrode_locs(patientID)['brainRegions_abbr']
    
    regionKeys = brainRegions_abbr.keys()
    regInds = collections.OrderedDict()
    
    if isinstance(abbrs, str):
        abbrs = [aa.strip() for aa in abbrs.split(',')]
    
    for abbr in abbrs:
        
        subset = [rr for rr in regionKeys if abbr in rr]
        
        for ss in subset:
            regInds[ss] = pd.Index(electrodeDF['abbr']).get_loc(ss)
    
    
    return regInds



def get_crossregion_pairs(abbrs, abbrInds):
    """
    Get pairwise combinations of all regions/electrodes defined by abbrs
    
    Inputs:
        - abbrs is a list or string (comma separated) of abbreviations of brain regions (see electrode_locs for more info)
          eg. abbrs = ['AM', 'HPC']  or 'AM, HPC', to choose amygdala and hippocampus electrodes
        - abbrInds is a dictionary defined by {'abbr': index/electrode list} (output of get_region_index)
    """
    
    if isinstance(abbrs, str):
        abbrs = utils.str_to_list(abbrs)
            
    allInds = []
    allRegs = []
    
    for reg in abbrs:
        keys = [key for key, ind in abbrInds.items() if reg in key]
        indices = [ind for key, ind in abbrInds.items() if reg in key]
        allRegs.append(keys)
        allInds.append(indices)
    
    pairNames = utils.get_pairs(allRegs[0], allRegs[1])
    pairInds = []
    
    if type(allInds[0][0]) == int and type(allInds[1][0]) == int:
        pairInds = utils.get_pairs(allInds[0], allInds[1])
        
    elif type(allInds[0][0]) == list and type(allInds[1][0]) == list:
        for ll1 in allInds[0]:
            for ll2 in allInds[1]:
                pairInds.append(utils.get_pairs(ll1, ll2))
    
    allPairs = collections.OrderedDict()
    for pairName, pairInd in zip(pairNames, pairInds):
         allPairs[pairName] = pairInd
    
    return allPairs
