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

from GeneralTools import *

class Merger(object):
    #CONTROL_AXES    = ['Learning Steps','Return','Time(s)','Features','Steps','Terminal','Episodes','Discounted Return']
    CONTROL_AXES    = ['Learning Steps','Return','Time(s)','Features','Steps','Terminal','Episodes']
    PE_AXES         = ['Iterations','Features','Error','Time(s)']
    #MDPSOLVER_AXES  = ['Bellman Updates', 'Return', 'Time(s)', 'Features', 'Steps', 'Terminal', 'Iterations', 'Discounted Return', 'Iteration Time']
    MDPSOLVER_AXES  = ['Bellman Updates', 'Return', 'Time(s)', 'Features', 'Steps', 'Terminal', 'Discounted Return', 'Iterations']
    prettyText = 1 #Use only if you want to copy paste from .txt files otherwise leave it to 0 so numpy can read such files.
    def __init__(self,paths, labels = None, output_path = None, colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple'], styles = ['o', 'v', '8', 's', 'p', '*', '<','h', '^', 'H', 'D',  '>', 'd'], markersize = 5, bars=1, legend = False, maxSamples = inf, minSamples = 1, getMAX = 0):
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        self.means                  = []
        self.std_errs               = []
        self.bars                   = bars  #Draw bars?
        self.colors                 = colors
        self.styles                 = styles
        self.markersize             = markersize
        self.legend                 = legend
        self.maxSamples             = maxSamples # In case we want to use less than available number of samples
        self.minSamples             = minSamples # Directories with samples less than this value will be ignored.
        self.samples                = []         # Number of Samples for each experiments
        self.useLastDataPoint       = False     # By default assume all data has the same number of points along the X axis.
        self.getMAX                 = getMAX    # Instead of mean of all experiments find the best performance among them
        self.output_path            = output_path
        self.labels                 = labels
        #Extract experiment paths by finding all subdirectories in all given paths that contain experiment.
        self.extractExperimentPaths(paths)
        
        #Setup the output path if it has not been defined:
        if output_path == None:
            self.output_path = '.' if len(paths) > 1 else paths[0]

        #Setup Labels if not defined
        if labels == None: self.labels = [self.extractLabel(p) for p in self.exp_paths]
        #print "Experiment Labels: ", self.labels

        self.exp_num                = len(self.exp_paths)
        self.means                  = []
        self.std_errs               = []
        if not isOnCluster():
            self.fig                    = pl.figure(1)
        self.datapoints_per_graph   = None # Number of datapoints to be shown for each graph (often this value is 10 corresponding to 10 performance checks)
        if len(self.exp_paths) == 0:
            print "No directory found with at least %d result files at %s." % (self.minSamples, paths)
            return
        for exp in self.exp_paths:
            means, std_errs, samples = self.parseExperiment(exp)
            self.means.append(means)
            self.std_errs.append(std_errs)
            self.samples.append(samples)
    def extractExperimentPaths(self,paths):
        self.exp_paths = []
        for path in paths:
            for p in os.walk(path):
                dirname = p[0]
                if self.hasResults(dirname):
                   self.exp_paths.append(dirname)
        if self.labels == None:
            self.exp_paths  = sorted(self.exp_paths)
    def extractLabel(self,p):
        # Given an experiment directoryname it extracts a reasonable label
        tokens = p.split('-')
        if len(tokens) > 1:
            return tokens[-2]+'-'+tokens[-1]
        else:
            return tokens[-1]
    def parseExperiment(self,path):
        # Parses all the files in form of <number>-results.txt and return
        # two matrices corresponding to mean and std_err
        # If the number of points along the x axis is not identical across all runs and experiments it returns the last data point for all experiments
        if '/' in path: 
            _,_,fringeDirName = path.rpartition('/') 
        else:
             fringeDirName = path
             
        files       = glob.glob('%s/*-results.txt'%path)
        samples_num = len(files)
        if self.maxSamples < samples_num:
            print "%d/%d Samples <= %s" % (samples_num, self.maxSamples,fringeDirName)
            samples_num = self.maxSamples
        else:
            print "%d Samples <= %s" % (samples_num, fringeDirName)
            
        #read the first file to initialize the matricies
        matrix      = readMatrixFromFile(files[0])
        rows, cols  = matrix.shape
        if rows == len(self.PE_AXES):
            self.ResultType = 'Policy Evaluation'
            self.AXES = self.PE_AXES
        elif rows == len(self.CONTROL_AXES):
            self.ResultType = 'RL-Control'
            self.AXES = self.CONTROL_AXES
        else:
            self.ResultType = 'MDPSolver'
            self.AXES = self.MDPSOLVER_AXES
            
        if not self.useLastDataPoint:
            samples     = zeros((rows,cols,samples_num))
            for i,f in enumerate(files):
                if i == samples_num: break
                M = readMatrixFromFile(files[i])
                #print M.shape
                try:
                    samples[:,:cols,i] = M
                except:
                    print "Unmatched number of points along X axis: %d != %d" % (cols,M.shape[1])
                    print "Switching to last data point mode for all experiments"
                    self.useLastDataPoint = True
                    break
        if self.useLastDataPoint:
            samples     = zeros((rows,1,samples_num))
            for i,f in enumerate(files):
                if i == samples_num: break
                M = readMatrixFromFile(files[i])
                samples[:,:,i] = M[:,-1].reshape((-1,1)) # Get the last column of the matrix

        _,self.datapoints_per_graph,_ = samples.shape
        if self.getMAX:
            return amax(samples,axis=2),std(samples,axis=2)/sqrt(samples_num), samples_num
        else:
            return mean(samples,axis=2),std(samples,axis=2)/sqrt(samples_num), samples_num
    def showLast(self,Y_axis = None):
        # Prints the last performance of all experiments
        if Y_axis == None: Y_axis = 'Error' if self.ResultType == 'Policy Evaluation' else 'Return'
        y_ind = self.AXES.index(Y_axis)
        M = array([M[y_ind,-1] for M in self.means])
        V = array([V[y_ind,-1] for V in self.std_errs])
        #os.system('clear');
        print "=========================="
        print "Last Performance+std_errs"
        print "=========================="
        for i,e in enumerate(self.exp_paths):
            print "%s:\t%0.3f+%0.3f based on %d Samples" % (e,M[i],V[i],self.samples[i])
    def bestExperiment(self,Y_axis = None, mode = 0):
        # Returns the experiment with the best final results
        # Best is defined in two settings:
        # Mode 0: Final Mean+variance
        # Mode 1: Area under the curve
        if Y_axis == None: Y_axis = 'Error' if self.ResultType == 'Policy Evaluation' else 'Return'
        y_ind = self.AXES.index(Y_axis)
        if mode == 0:
            M = array([M[y_ind,-1] for M in self.means])
            V = array([V[y_ind,-1] for V in self.std_errs])
            best_index = argmax(M+V)
            best = max(M+V)
        else:
            M = array([M[y_ind,:] for M in self.means])
            S = sum(M,axis=1)
            print M
            print S
            best = max(S)
            best_index = argmax(S)
        return self.exp_paths[best_index], best
    def plot(self,Y_axis = None, X_axis = None):
        #Setting default values based on the Policy Evaluation or control
        if Y_axis == None: Y_axis = 'Error' if self.ResultType == 'Policy Evaluation' else 'Return'
        if X_axis == None:
            if self.ResultType == 'RL-Control':
                X_axis = 'Learning Steps'
            elif self.ResultType == 'MDPSolver':
                X_axis = 'Time(s)'
            else:
                X_axis = 'Iterations'

        if not isOnCluster(): self.fig.clear()
        min_ = +inf
        max_ = -inf

        #See if requested axes are reasonable
        if Y_axis in self.AXES:
            y_ind = self.AXES.index(Y_axis)
        else:
            print 'unknown Y_axis = %s', Y_axis
            print 'Allowed values:'
            print self.AXES
        if X_axis in self.AXES:
            x_ind = self.AXES.index(X_axis)
        else:
            print 'unknown X_axis = %s', X_axis
            print 'Allowed values:'
            print self.AXES

        Xs      = zeros((self.exp_num,self.datapoints_per_graph))
        Ys      = zeros((self.exp_num,self.datapoints_per_graph))
        Errs    = zeros((self.exp_num,self.datapoints_per_graph))

        for i in arange(self.exp_num):
            X   = self.means[i][x_ind,:]
            Y   = self.means[i][y_ind,:]
            Err = self.std_errs[i][y_ind,:]
            if not isOnCluster():
                if len(X) == 1 and self.bars:
                    pl.errorbar(X, Y, yerr=Err, marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7,color = self.colors[i%len(self.colors)],markersize = self.markersize, label = self.labels[i])
                    max_ = max(max(Y+Err),max_); min_ = min(min(Y-Err),min_)
                else:
                    pl.plot(X,Y,linestyle ='-', marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7,color = self.colors[i%len(self.colors)],markersize = self.markersize, label = self.labels[i])
                    if self.bars:
                        pl.fill_between(X, Y-Err, Y+Err,alpha=.1, color = self.colors[i%len(self.colors)])
                        max_ = max(max(Y+Err),max_); min_ = min(min(Y-Err),min_)
                    else:
                        max_ = max(Y.max(),max_); min_ = min(Y.min(),min_)
            Xs[i,:]     = X
            Ys[i,:]     = Y
            Errs[i,:]   = Err

        if not isOnCluster():
            if self.legend:
                #pl.legend(loc='lower right',b_to_anchor=(0, 0),fancybox=True,shadow=True, ncol=1, mode='')
                self.legend = pl.legend(fancybox=True,shadow=True, ncol=1, frameon=True,loc=(1.03,0.2))
                #pl.axes([0.125,0.2,0.95-0.125,0.95-0.2])
            pl.xlim(0,max(Xs[:,-1])*1.02)
            if min_ != max_: pl.ylim(min_-.1*abs(max_-min_),max_+.1*abs(max_-min_))
            X_axis_label = r'$\|A\theta - b\|$' if X_axis == 'Error' else X_axis
            Y_axis_label = r'$\|A\theta - b\|$' if Y_axis == 'Error' else Y_axis
            pl.xlabel(X_axis_label,fontsize=16)
            pl.ylabel(Y_axis_label,fontsize=16)
        self.save(Y_axis,X_axis,Xs,Ys,Errs)
        if not isOnCluster and self.legend:
                # This is a hack so we can see it correctly during the runtime
                pl.legend(loc='lower right',fancybox=True,shadow=True, ncol=1, mode='')
    def save(self,Y_axis,X_axis,Xs,Ys,Errs):
        fullfilename = self.output_path + '/' +Y_axis+'-by-'+X_axis
        checkNCreateDirectory(fullfilename)
        # Store the numbers in a txt file
        f = open(fullfilename+'.txt','w')
        for i in range(self.exp_num):
            # Print in the standard error:
            print "======================"
            print "Algorithm: ", self.exp_paths[i]
            print "======================"
            print X_axis +': ' + pretty(Xs[i,:],'%0.0f')
            print Y_axis +': ' + pretty(Ys[i,:])
            print 'Standard-Error: ' + pretty(Errs[i,:])
            if self.prettyText:
                f.write("======================\n")
                f.write("Algorithm: " + self.labels[i] +"\n")
                f.write("======================\n")
                f.write(X_axis +': ' + pretty(Xs[i,:])+'\n')
                f.write(Y_axis +': ' + pretty(Ys[i,:])+'\n')
                f.write('Standard-Error: ' + pretty(Errs[i,:])+'\n')
            else:
                savetxt(f,Xs[i,:], fmt='%0.4f', delimiter='\t')
                savetxt(f,Ys[i,:], fmt='%0.4f', delimiter='\t')
                savetxt(f,Errs[i,:], fmt='%0.4f', delimiter='\t')
        f.close()
        # Save the figure as pdf
        if not isOnCluster():
            if self.legend:
                self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1,bbox_extra_artists=(self.legend,), bbox_inches='tight')
            else:
                self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1, bbox_inches='tight')
            print "==================\nSaved Outputs at\n1. %s\n2. %s" % (fullfilename+'.txt',fullfilename+'.pdf')
    def hasResults(self,path):
        return len(glob.glob(os.path.join(path, '*-results.txt'))) >= self.minSamples
