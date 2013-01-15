#!/usr/bin/python

import os
import sys


def main(jobID=-1,              # Used as an indicator for each run of the algorithm
         PROJECT_PATH ='',      # Path to store the results. Notice that a directory is automatically generated within this directory based on the selection of domain,agent,representation, 
         SHOW_FINAL_PLOT=0):    # Draw the final plot when the run is finished? Automatically set to False if jobID == -1
	res = open('1-result'+'.txt','w')
	res.write('test')
	res.close()
	out = open('1-out'+'.txt','w')
	out.write('test')
	out.close()

if __name__ == '__main__':
     if len(sys.argv) == 1: #Single Run
         print os.getpid()
         main(jobID = 1,PROJECT_PATH = 'Results/Temp',SHOW_FINAL_PLOT = True)
     else: # Batch Mode through command line
         print os.getpid()
	 print 'woot'
         main(int(sys.argv[1]),sys.argv[2])
