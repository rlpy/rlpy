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
from copy import deepcopy #
import os, sys, multiprocessing
from os import path
from decimal import Decimal
# If running on an older version of numpy, check to make sure we have defined all required functions.
import numpy # We need to be able to reference numpy by name


class PriorityQueueWithNovelty():
    # This is a priority queue where it is sorted based on priority and then then novelty of elements
    # First Order: The Lower the priority the better
    # Second Order: The newer the item the better
    # Example:
    # put (1,<O1>)
    # put (2,<O2>)
    # put (1,<O3>)
    # put (10,<O4>)
    # => multiple get() : O3,O1,O2,O4
    # Adopted from http://stackoverflow.com/questions/9289614/how-to-put-items-into-priority-queues
    def __init__(self):
        self.h = []
        self.counter = 0
    def push(self, priority, item):
        heappush(self.h,(priority, self.counter, item))
        self.counter -= 1
    def pop(self):
        _, _, item = heappop(self.h)
        return item
    def empty(self):
        return len(self.h) == 0
    def toList(self):
        temp = list(self.h)
        return [heappop(temp)[2] for i in range(len(temp))]
    def show(self):
        temp = list(self.h)
        for i in range(len(temp)):
            p,c,x = heappop(temp)
            print "Priotiry = %d, Novelty = %d, Obj = %s" % (p,c,str(x))
 