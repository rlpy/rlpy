#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard March 21th 2013 at MIT #
######################################################
# Shows the average of performance of each directory
 
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
merger.plot()