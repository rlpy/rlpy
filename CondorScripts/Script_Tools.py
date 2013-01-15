#!/usr/bin/python
import os, sys, time, re, string, pprint, re

#Colors 
PURPLE  = '\033[95m'
BLUE    = '\033[94m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
RED     = '\033[91m'
NOCOLOR = '\033[0m'

COMPLETED_COLOR = GREEN
RUNNING_COLOR   = RED
TOTAL_COLOR     = BLUE
RESUMING_COLOR  = PURPLE

PARALLEL_SUBDIR     = 'Results/' # Where each jobs are supposed to be stored
USERNAME            = 'agf'
FINALFILE           = 'results.txt'
GRAPH_DIR           = ' ~/Graphs/FeatureDiscovery/'
PYTHON_PATH          = ' /usr/bin/python'
FINISHED_RUNS_NUM   = 5   # Number of runs which is counted as a finished run for testing a learning rate

import os

def logKey(log):
    timeRE     = re.compile('R\[[0-9]*:[0-9]*:[0-9]*')
    temp1       = timeRE.search(log)
    if not temp1:
        return sys.maxint
    else:
        temp1 = temp1.group(0)[2:] 
        
    [h1,m1,s1] = re.split(':+',temp1)
#    print h1, m1, s1
    h1 = eval(h1)    
    m1 = eval(m1)    
    s1 = eval(s1)    
    
    return h1*10000+m1*100+s1

def sortLog(logs):
    return sorted(logs,key=logKey)

if __name__ == '__main__':
    print('************** Sort Log Test **********************');    
    logs = ['633: E[0:00:01]-R[0:00:07]: Return=-1.01, Steps=12, Features = 10',
            '1283: E[0:00:02]-R[0:00:06]: Return=-1.01, Steps=16, Features = 10',
            '2582: E[0:00:04]-R[0:00:04]: Return=-0.04, Steps=40, Features = 10',
            '1932: E[0:00:03]-R[0:00:05]: Return=-0.04, Steps=40, Features = 10',
            '3892: E[0:00:06]-R[0:00:02]: Return=-0.04, Steps=40, Features = 10',
            '3227: E[0:00:05]-R[0:00:03]: Return=-0.04, Steps=40, Features = 10',
            '4548: E[0:00:07]-R[0:00:01]: Return=-0.04, Steps=40, Features = 10']
    
    for log in logs:
        print "%s" % log
        
    logs = sortLog(logs);
    print "after sort:"
    for log in logs:
        print "%s" % log

# This value is used to avoid actually doing anything, so we can check the program
