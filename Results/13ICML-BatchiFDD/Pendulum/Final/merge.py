#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

import sys, os
#Add all paths
path = '.'
while not os.path.exists(path+'/Tools.py'):
    path = path + '/..'
sys.path.insert(0, os.path.abspath(path))
from Tools import *

#paths = ['.'] 
paths = [
         'Pendulum_InvertedBalance-IndependentDiscretization-10000',
         'Pendulum_InvertedBalance-BEBF-10000-0.2',
         'Pendulum_InvertedBalance-iFDD-10000-0.6',
         'Pendulum_InvertedBalance-OMPTD-10000-0.6'
        ] 

#paths = [
#         'Pendulum_InvertedBalance-BEBF-20000-0.25',
#         'Pendulum_InvertedBalance-BEBF-20000-0.3'
#         ]
labels      = ['Initial','BEBF','iFDD','OMPTD']
labels      = []
colors      = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple','b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles      = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd','o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
MarkerSize  = 15
Legend      = True

merger = Merger(paths,labels=labels, colors = colors, styles= styles, markersize = MarkerSize, legend = Legend)
pl.ioff()
#print mergedData.means[0].shape
#merger.plot('Return')
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


