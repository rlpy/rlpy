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

if __name__ == '__main__':
    
    paths = ['.']
    merger = Merger(['.'],showSplash=False,getMAX = 1)
    if len(sys.argv) == 1:
        merger.showLast()
    else:
        merger.showLast(sys.argv[1])
