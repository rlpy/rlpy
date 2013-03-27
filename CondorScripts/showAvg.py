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

def searchNShowAvg(path,Y_axis = None):
    if os.path.exists('main.py'):
        merger = Merger([path])
        if Y_axis:
            merger.showLast(sys.argv[1])
        else:
            merger.showLast()
    else:
        for d in os.listdir(path):
            if os.path.isdir(d) and d[0] != '.':
                #print "Looking into: %s" % d
                searchNShowAvg(path+'/'+d)
        
if __name__ == '__main__':
    if len(sys.argv) == 1:
        searchNShowAvg('.')
    else:
        searchNShowAvg('.',sys.argv[1])
