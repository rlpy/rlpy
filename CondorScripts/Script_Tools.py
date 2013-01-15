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

PARALLEL_SUBDIR     = 'Results/Parallel/' # Where each jobs are supposed to be stored
USERNAME            = 'agf'
FINALFILE           = 'Results.mat'
GRAPH_DIR           = ' ~/Graphs/FeatureDiscovery/'
MERGE_PATH          = GRAPH_DIR + 'merge.m'
MERGE_LEARNING_PATH = GRAPH_DIR + 'mergeLearning.m'
MERGE_VERTICAL_PATH = GRAPH_DIR + 'mergeVertical.m'
JAVA_OPT_PATH       = GRAPH_DIR + 'java.opts'
MATLAB_EXE          = ' /afs/csail.mit.edu/system/common/matlab/2010a/bin/matlab -nodisplay -nosplash'
FINISHED_RUNS_NUM   = 5   # Number of runs which is counted as a finished run for testing a learning rate
LEARNING_DIRS       = ['a.01N100',
                       'a.01N1000',
                       'a.01N1000000',
                       'a.1N100',
                       'a.1N1000',
                       'a.1N1000000',
                       'a1N100',
                       'a1N1000',
                       'a1N1000000'
                       ]

import os

def logKey(log):
    timeRE     = re.compile('[0-9]*:[0-9]*:[0-9]*')
    temp1       = timeRE.search(log)
    if not temp1:
        return sys.maxint
    else:
        temp1 = temp1.group(0) 

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
    logs = ['>>> [70.8%] [0:3:32] <<0:00:30>> Steps:[7076/10000]Return: +0.00 Steps:320',
            '>>> [74.2%] [0:0:21] <<0:00:31>> Steps:[7418/10000]Return: +0.00 Steps:342',
            '>>> [76.1%] [0:1:2] <<0:00:31>> Steps:[7606/10000]Return: +0.00 Steps:188']
    
    for log in logs:
        print "%s" % log
        
    logs = sortLog(logs);

    for log in logs:
        print "%s" % log

# This value is used to avoid actually doing anything, so we can check the program
