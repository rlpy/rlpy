#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard March 20th 2013 at MIT #
######################################################
# Find the best parameter setting and run it for 30 times.
# Best is defined as last data sample mean + variance.
# Assumption: results for each parameter setting is in a separate directory
# 1. Find the mean and variance of the final set of points
# 2. run he script runall in the directory with the best results
 
import sys, os
#Add all paths
path = '.'
print 'looking for Tools.py'
while not os.path.exists(path+'/RL-Python/Tools.py'):
    path = path + '/..'
    print path
path += '/RL-Python'
sys.path.insert(0, os.path.abspath(path))
from Tools import *

paths = ['.']
merger = Merger(paths)
bestExp, bestValue = merger.bestExperiment(mode = 1)
if bestExp:
    print "======================"
    print "Best Experiment: %s with Value=%0.4f" % (bestExp, bestValue)
    print "======================"
    os.chdir(bestExp)
    os.system('runall.py')
