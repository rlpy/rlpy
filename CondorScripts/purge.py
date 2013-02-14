#!/usr/bin/python
# Functions used to remove all unfinished jobs
# Be careful if jobs are running on the server you should not run this command on the corresponding directory
# Alborz Geramifard 2009 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?
# It also remove CondorOutput
#Inputs:
# idir : Initial Directory

import os, sys, time, re, string
from Script_Tools import *

def searchNPurge(idir):
                    
        #print idir                    
        currentdir      = os.getcwd()
        os.chdir(idir) 

        if os.path.exists('main.py'):
            
            if idir != '.':
                print 'Experiment: '+ idir
            
            os.system("rm -rf CondorOutput")
            
            outjobs     = glob.glob('*-out.txt')
            resjobs     = glob.glob('*-results.txt')
            completed   = len(resjobs)
            purged      = 0
            for outjob in outjobs: 
                jobid,_,_ = outjob.rpartition('-')
                if not os.path.exists(jobid+'-results.txt'):
                    os.system("rm -r " + outjob)
                    os.system("rm -r " + 'CondorOutput/log/%s.log' % jobid)
                    os.system("rm -r " + 'CondorOutput/err/%s.err' % jobid)
                    os.system("rm -r " + 'CondorOutput/err/%s.out' % jobid)
                    print RED+">>> Purged Job # "+jobid+NOCOLOR
                    purged += 1        
            print "Completed Jobs:\t%d" % completed
            print "Purged Jobs:\t%d" % (purged)
            print "====================="
        else:
            for folder in os.listdir('.'):
                if os.path.isdir(folder) and not folder.startswith('.'):
                    searchNPurge(folder)
                    
        #Return to the directory we started at
        os.chdir(currentdir) 

if __name__ == '__main__':
    
    os.system('clear');

    print('***********************************************');    
    print('***************** Purge all unfinished ********');    
    print('***********************************************');     
    
    searchNPurge('.')   
    
