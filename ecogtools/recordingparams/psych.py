# 2015-07-06, LKirkby

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def IMS_timepoints(patientID, plotIMS=False, VAS=False):
    """
    IMS (immediate mood scaler) scores and timepoints for each patient

    (Excluding points within ~1.5hr of each other)
    """

    if patientID == 'EC77':
        IMS = [-13, -1, 5]
        IMStimes = [1417626792, 1417723978, 1417736009]

    elif (patientID == 'EC79'):
        IMS = [16, 2, -5, -9, 16, 23, 22, 25, 21, 18, 16]
        IMStimes = [1418776854, 1418839417, 1418939759, 1419019436, 1419034350,
                    1419124816, 1419133920, 1419198972, 1419206438, 1419211244, 1419269571]

    elif (patientID == 'EC80'):
        IMS = [-3, 0, 3, 2, 4, 2, 1, 1, -6, -3, -3, 4, 4, 0]
        IMStimes = [1421185973, 1421200903, 1421204014, 1421257148, 1421264070, 1421275813,
                    1421279198, 1421341357, 1421363803, 1421372925, 1421384269, 1421433702, 1421439952, 1421448046]

    elif patientID == 'EC82':
        IMS = [-14, -18, -14, -23, -19, -18, -24, -
               30, -28, -29, -26, -32, -40, -21, -29, -4]
        IMStimes = [1428358178, 1428367773, 1428378448, 1428411910, 1428427747,
                    1428434728, 1428454756, 1428497066, 1428509880, 1428519640, 1428547451, 1428583557, 1428589812, 1428606366, 1428616149, 1428618735]

    elif patientID == 'EC84':
        IMS = [-4, -8, -8, -10, -8, -1, -4, -
               15, -13, -9, -4, -4, -5, -10, -6, -7]
        IMStimes = [1429656044, 1429656978, 1429720312, 1429732937, 1429824898, 1429890813, 1429894556, 1430341433,
                    1430412507, 1430432405, 1430504276, 1430522359, 1430525011, 1430594427, 1430768763, 1430772708]

    elif patientID == 'EC108':
        IMS = [-2, -10, 34, 20, 11, 30, 48, 31, 34, 50]
        IMStimes = [1449700327, 1449710372, 1449756369, 1449776719, 1449784781,
                    1449959421, 1449968766, 1450024203, 1450053238, 1450108845]

    elif patientID == 'EC113':
        IMS = [34, 42, 44, 45, 44, 54, 56, 71, 72]
        IMStimes = [1454447451, 1454705720, 1454721185, 1454782742,
                    1454795482, 1454829343, 1454965583, 1454966177, 1454967179]

    elif patientID == 'EC122':
        IMS = [61, 66, 67, 67, 66, 63, 66, 67, 68, 68, 69, 69, 70]
        IMStimes = [1463517372, 1463541620, 1463584937, 1463610981, 1463672292, 1463713631,
                    1463861844, 1463875727, 1463892997, 1463929927, 1463956173, 1463972281, 1464017942]

    elif patientID == 'EC125':
        IMS = [-23, -16, -38, -18, -18, -43, -18, -15, -43, 10, -18]
        IMStimes = [1466969406, 1466978448, 1467051471, 1467076957, 1467090020,
                    1467139326, 1467149152, 1467156090, 1467163846, 1467218717, 1467225058]

    elif patientID == 'EC129':
        IMS = [-1, -22, -46, 29, 56, 48]
        IMStimes = [1470936740, 1470958057, 1471029155,
                    1471045617, 1471130593, 1471206462]

    elif patientID == 'EC131':
        IMS = [45, 34, 41, 29, 27, 32]
        IMStimes = [1472601632, 1472774119, 1472850835,
                    1472863544, 1473187584, 1473212557]

    elif patientID == 'EC133':
        IMS = [42, 6, 44, 32]
        IMStimes = [1473800052, 1473869679, 1473889342, 1474038847]

    elif patientID == 'EC136':
        IMS = [12, 35, 9, 28, 19]
        IMStimes = [1479245557, 1479266602, 1479313692, 1479333100, 1479353521]

    elif patientID == 'EC139':
        IMS = [58, 64, 60]
        IMStimes = [1481750966, 1481844536, 1482009293]

    elif patientID == 'EC142':
        IMS = [-25, -56, -59, -52, -47, -55]
        IMStimes = [1485906004, 1485960390, 1486050618,
                    1486058763, 1486079645, 1486129848]

    elif patientID == 'EC143':
        IMS = [21, 28, 2, 28]
        IMStimes = [1486590176, 1486680537, 1486753647, 1486758307]

    elif patientID == 'EC148':
        IMS = [7, 36, 38]
        IMStimes = [1491943907, 1491955762, 1492026177]

    elif patientID == 'EC153':
        IMS = [43, 35, 45]
        IMStimes = [1494974497, 1495142613, 1495252781]

    elif patientID == 'EC155':
        IMS = [-7, 30, 25]
        IMStimes = [1499884136, 1499995806, 1500002284]

    elif patientID == 'EC156':
        IMS = [53, 65, 67, 69, 70, 68, 69, 69, 72, 72]
        IMStimes = [1499813570, 1499879681, 1499964358, 1499969130, 1499987608,
                    1500049827, 1500059623, 1500239808, 1500332014, 1500336257]

    elif patientID == 'EC158':
        IMS = [-5, 35, 36, 2, 41, 54, 21]
        IMStimes = [1502998568, 1503010180, 1503017548,
                    1503028417, 1503072633, 1503080308, 1503183747]

    else:
        print '\tNo IMS time points for patient '+str(patientID)
        IMS = None
        IMStimes = None
        IMSdatetime = None
        plotIMS = False
        IMSdataframe = None

    # Normalized IMS:
    if patientID == 'EC77' or patientID == 'EC79' or patientID == 'EC80':
        IMSrange = [-69, 69]
    elif patientID == 'EC84':
        IMSrange = [-15, 15]
    else:
        IMSrange = [-72, 72]

    IMSnorm = [
        int(round(((float(II)-IMSrange[0])/(IMSrange[1]-IMSrange[0]))*100)) for II in IMS]

    if IMS != None:
        IMSdatetime = [datetime.datetime.fromtimestamp(
            timePoint) for timePoint in IMStimes]
        IMSdata = {'IMStimes': IMSdatetime, 'IMS': IMS, 'IMSnorm': IMSnorm}
        IMSdataframe = pd.DataFrame(data=IMSdata)

    # Plot IMS score over time if plotIMS flag is True:

    if plotIMS == True:
        # matplotlib date format object
        tAxisPlotIMS = [mdates.date2num(timeObj) for timeObj in IMSdatetime]
        tformatmajor = mdates.DateFormatter('%y:%m/%d %H:%M:%S')
        tformatminor = mdates.DateFormatter('%H:%M:%S')

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot_date(tAxisPlotIMS, IMS, 'k.--', markersize=15)
        ax.xaxis.set_major_formatter(tformatmajor)
        ax.xaxis.set_minor_formatter(tformatminor)
        fig.autofmt_xdate()
        plt.grid()
        plt.title('IMS for '+patientID)
        plt.ylabel('IMS score')

    return IMSdataframe


def psych_scores(patientID):

    if patientID == 'EC77':
        BDI = 5
        BAI = 7
        PHQ9 = 3
        GAD7 = 3
        Rum = 41
    elif patientID == 'EC79':
        BDI = 28
        BAI = 8
        PHQ9 = 12
        GAD7 = 11
        Rum = 57
    elif patientID == 'EC80':
        BDI = 8
        BAI = 2
        PHQ9 = 2
        GAD7 = 6
        Rum = 56
    elif patientID == 'EC82' or patientID == 'EC82b':
        BDI = None
        BAI = None
        PHQ9 = 13
        GAD7 = 10
        Rum = 55
    elif patientID == 'EC84':
        BDI = None
        BAI = None
        PHQ9 = 15
        GAD7 = 15
        Rum = 65
    elif patientID == 'EC108':
        BDI = None
        BAI = None
        PHQ9 = 10
        GAD7 = 10
        Rum = 54
    elif patientID == 'EC113':
        BDI = 1
        BAI = 3
        PHQ9 = 2
        GAD7 = 3
        Rum = 29
    elif patientID == 'EC122':
        BDI = 6
        BAI = 6
        PHQ9 = 5
        GAD7 = 3
        Rum = 25
    elif patientID == 'EC125':
        BDI = 28
        BAI = 16
        PHQ9 = 12
        GAD7 = 12
        Rum = 58
    elif patientID == 'EC129':
        BDI = 8
        BAI = 6
        PHQ9 = 4
        GAD7 = 7
        Rum = 37
    elif patientID == 'EC131':
        BDI = 15
        BAI = 19
        PHQ9 = 9
        GAD7 = 5
        Rum = 32
    elif patientID == 'EC133':
        BDI = None
        BAI = None
        PHQ9 = 3
        GAD7 = 6
        Rum = 37
    elif patientID == 'EC136':
        BDI = 32
        BAI = 5
        PHQ9 = 10
        GAD7 = 5
        Rum = 56
    elif patientID == 'EC139':
        BDI = 32
        BAI = 9
        PHQ9 = 4
        GAD7 = 11
        Rum = None
    elif patientID == 'EC142':
        BDI = 20
        BAI = 9
        PHQ9 = 15
        GAD7 = 12
        Rum = 63
    elif patientID == 'EC143':
        BDI = 12
        BAI = 26
        PHQ9 = 10
        GAD7 = 8
        Rum = 22
    elif patientID == 'EC148':
        BDI = None
        BAI = None
        PHQ9 = 1
        GAD7 = 0
        Rum = None
    elif patientID == 'EC153':
        BDI = 33
        BAI = 42
        PHQ9 = 2
        GAD7 = 1
        Rum = None
    elif patientID == 'EC155':
        BDI = 7
        BAI = 4
        PHQ9 = 12
        GAD7 = 7
        Rum = None
    elif patientID == 'EC156':
        BDI = 7
        BAI = 6
        PHQ9 = 4
        GAD7 = 3
        Rum = None
    elif patientID == 'EC158':
        BDI = 28
        BAI = 43
        PHQ9 = 6
        GAD7 = 12
        Rum = None
    else:
        print('No psych data for '+patientID)
        BDI = None
        BAI = None
        PHQ9 = None
        GAD7 = None
        Rum = None

    return {'BDI': BDI, 'BAI': BAI, 'PHQ9': PHQ9, 'GAD7': GAD7, 'Rum': Rum}
