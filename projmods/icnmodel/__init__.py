from os.path import dirname, basename, isfile
import glob

modules = glob.glob(dirname(__file__)+'/*.py')

__all__ = [basename(ff)[:-3] for ff in modules if isfile(ff)]

try:
    from . import * 
except:
    import icnmodel

from icnmodel import *

del ff
del dirname
del basename
del isfile
del glob
del modules
