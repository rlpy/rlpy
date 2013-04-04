#!/usr/bin/python
######################################################
# Developed by Alborz Geramiard March 21th 2013 at MIT #
######################################################
# Shows the average of performance of each directory
 
import sys, os
from Script_Tools import *
path = findRLRoot()
sys.path.insert(0, path)
from Tools import *

def searchNShowAvg(path,Y_axis = None):
    if os.path.exists(path+'/main.py'):
        merger = Merger([path])
        if Y_axis:
            merger.showLast(sys.argv[1])
        else:
            merger.showLast()
    else:
        for d in os.listdir(path):
            if os.path.isdir(d) and d[0] != '.':
                print "Looking into: %s" % d
                searchNShowAvg(path+'/'+d)
        
if __name__ == '__main__':
#    if len(sys.argv) == 1:
#        searchNShowAvg('.')
#    else:
#        searchNShowAvg('.',sys.argv[1])
    merger = Merger(['.'],showSplash=False)
    if len(sys.argv) > 1:
        merger.showLast(sys.argv[1])
    else:
        merger.showLast()
        
