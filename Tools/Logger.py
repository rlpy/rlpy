def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True

from multiprocessing import Pool
from operator import *
from numpy  import *
from Config import *
import itertools
import platform
import pdb
import os

from GeneralTools import *
#For condor use
os.environ['HOME'] = HOME_DIR  # matplotlib attempts to write to a condor directory in "~" which it doesn't own; have it write to tmp instead, common solution on forums
os.environ['MPLCONFIGDIR'] = os.environ['HOME']

if module_exists('matplotlib'):
    from matplotlib import pylab as pl
    from matplotlib import mpl,rc,colors
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    import matplotlib.cm as cm
    from matplotlib.mlab import rk4
    from matplotlib import lines
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import lines # for plotting lines in pendulum and PST
    from matplotlib.mlab import rk4 # for integration in pendulum
    from matplotlib.patches import ConnectionStyle # for cartpole
    pl.ion()
else:
    print 'matplotlib is not available => No Graphics'

if module_exists('networkx'):
    import networkx as nx
else:
    'networkx is not available => No Graphics on SystemAdmin domain'

if module_exists('sklearn'):
    from sklearn import svm
    from sklearn.gaussian_process import GaussianProcess
else:
    'sklearn is not available => No BEBF representation available'
import glob
from scipy import stats
from scipy import misc
from scipy import linalg
from scipy.sparse import linalg as slinalg
from scipy import sparse as sp
from time import *
from hashlib import sha1
import datetime, csv
from string import *
#from Sets import ImmutableSet
from itertools  import *
from heapq import *
from copy import deepcopy
import os, sys, multiprocessing
from os import path
from decimal import Decimal
# If running on an older version of numpy, check to make sure we have defined all required functions.
import numpy # We need to be able to reference numpy by name


class Logger(object):
    buffer = ''         # You can print into a logger without initializing its filename. Whenever the filename is set, the buffer is flushed to the output.
    filename = ''
    def setOutput(self,filename):
        if self.filename != '':
            print "Warning: logger has been initialized to another file: %s. The rest of output will be in %s" % (self.filename, filename)
        self.filename = filename
        checkNCreateDirectory(filename)
        f = open(self.filename,'w')
        f.close()
    def log(self,str):
    # Print something both in output and in a file
        print str
        self.buffer += str +'\n'
        if self.filename != '':
            f = open(self.filename,'a')
            f.write(self.buffer)
            f.close()
            self.buffer = ''
    def line(self):
        self.log(SEP_LINE)
        
#Colors
PURPLE  = '\033[95m'
BLUE    = '\033[94m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
RED     = '\033[91m'
NOCOLOR = '\033[0m'
RESEDUAL_THRESHOLD = 1e-7
REGULARIZATION = 1e-6
FONTSIZE = 15
SEP_LINE = "="*60