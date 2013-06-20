#!/usr/bin/env python
######################################################
# Developed by Alborz Geramiard March 21th 2013 at MIT #
######################################################
# Shows the average of time to reach 95% of the last performance
 
import sys, os
from Script_Tools import *
path = findRLRoot()
sys.path.insert(0, path)
from Tools import *

if __name__ == '__main__':
    merger = Merger(['.'],showSplash=False)
    merger.showTime95()
        
