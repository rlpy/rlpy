#!/usr/bin/env python
"""Merge several results runs into one plot."""

# Merge multiple results of several algorithms and show them on one plot
# Copy this file in the parent directory where the runs of all of your algorithms reside.
# e.g:
# /Project
# /Project/Algorithm1
# /Project/Algorithm2
# /Project/Algorithm3
# /Project/Algorithm4
# copy merge.py in /Project
# OR set the path to the correct location
# If you want to select a set of runs you can pass a list of paths that you would like to be considered.

#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.' 
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Alborz Geramifard"

paths = ['.']
#paths = ['Results/Example_Project']
#paths = ['Results/Example_Project'] 
#paths = ['R    esults/13ICML-BatchiFDD/PST/PST-iFDD-50000-65'] 

labels              = []
colors              = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles              = ['o', 'v', '8', 's', 'p', '*', '<','h', '^', 'H', 'D',  '>', 'd']
MarkerSize          = 7    
Legend              = True  # Draw legends?
maxSamples          = inf   # Maximum number of samples to be loaded from the directory. If set inf it will use all of them
minSamples          = 1     # Minimum number of samples required to include a directory for plotting 
useLastDatapoint    = False # IF true, only uses the result obtained at the final iteration/timestep [

merger = Merger(paths, labels=labels, colors = colors, styles= styles, markersize = MarkerSize, legend = Legend, maxSamples = maxSamples, minSamples = minSamples)
pl.ioff()

#Extract experiment paths by finding all subdirectories in all given paths that contain experiment.
# merger.extractExperimentPaths(paths)
# merger.computeStatistics()

# FOR Control
#######################
#merger.plot('Return')
#merger.plot('Return','Features')
#merger.plot('Return','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Steps','Features')
#merger.plot('Steps')
#merger.plot('Steps','Learning Steps')
#merger.plot('Steps','Episodes')
#merger.plot('Steps','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Features','Time(s)')
#merger.plot('Terminal')
# FOR POLICY EVALUTION
#######################
#merger.plot("Error",'Iterations')
#merger.plot("Error",'Features')
#merger.plot("Error",'Time(s)')
# FOR MDP Solvers
#######################
#merger.plot('Return')
#merger.plot('Return','Features')
#merger.plot('Return','Time(s)')

merger.plot() #Plot the default Y-Axis and X-Axis 

if not isOnCluster():
    pl.show()


