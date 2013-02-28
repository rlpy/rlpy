#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
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

#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

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
#paths = ['Results/Example_Project']
#paths = ['Results/Example_Project'] 
#paths = ['R    esults/13ICML-BatchiFDD/PST/PST-iFDD-50000-65'] 

labels              = []
colors              = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles              = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
MarkerSize          = 10    
Legend              = True  # Draw legends?
maxSamples          = inf   # Maximum number of samples to be loaded from the directory. If set inf it will use all of them
 
merger = Merger(paths,labels=labels, colors = colors, styles= styles, markersize = MarkerSize, legend = Legend, maxSamples = maxSamples)
pl.ioff()
#print mergedData.means[0].shape
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
 
merger.plot() #Works both with control and PE plotting the most common output

if not isOnCluster():
    pl.show()


