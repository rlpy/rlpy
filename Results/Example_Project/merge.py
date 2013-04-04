#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

import sys, os
#Add all paths
RL_PYTHON_ROOT = os.environ.get('RL_PYTHON_ROOT')
if (RL_PYTHON_ROOT == None):
    print 'Could not get environment variable RL_PYTHON_ROOT: \
    \nplease re-run installer script or see FAQ.txt. \nExiting.'
    sys.exit()

sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *

paths = ['.'] 
#paths = [
#         'Pendulum_InvertedBalance-IndependentDiscretization-20000',
#         'Pendulum_InvertedBalance-BEBF-20000-0.2',
#         'Pendulum_InvertedBalance-iFDD-20000-0.3'
#        ] 

#paths = [
#         'Pendulum_InvertedBalance-BEBF-20000-0.25',
#         'Pendulum_InvertedBalance-BEBF-20000-0.3'
#         ]
#labels      = ['Initial','BEBF','iFDD']
labels      = []
colors      = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple','b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles      = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd','o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
MarkerSize  = 15
Legend      = True

merger = Merger(paths,labels=labels, colors = colors, styles= styles, markersize = MarkerSize, legend = Legend)
pl.ioff()
#print mergedData.means[0].shape
merger.plot('Return')
#merger.plot('Steps','Time(s)')
merger.plot('Steps','Features')
#merger.plot('Steps')
#merger.plot('Steps','Learning Steps')
#merger.plot('Steps','Episodes')
#merger.plot('Steps','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Features','Time(s)')
#merger.plot('Terminal')
pl.show()


