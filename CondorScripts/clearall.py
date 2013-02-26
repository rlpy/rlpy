#!/usr/bin/python
# Functions used to remove all jobs from directories
# Be careful if jobs are running on the server you should not run this command on the corresponding directory
# Alborz Geramifard 2009 MIT
# Assumes linux machine just for clear screen! Why do you want to run it on something else anyway?
# It also remove CondorOutput
#Inputs:
# idir : Initial Directory

import os, sys, time, re, string
from Script_Tools import *

def searchNClear(idir):
                    
        #print idir                    
        currentdir      = os.getcwd()
        os.chdir(idir) 

        if os.path.exists('main.py'):
            
            if idir != '.':
                print 'Experiment: '+ idir
            
            os.system("rm -rf CondorOutput")
            os.system("rm -rf *.txt")
        else:
            for folder in os.listdir('.'):
                if os.path.isdir(folder) and not folder.startswith('.'):
                    searchNPurge(folder)
                    
        #Return to the directory we started at
        os.chdir(currentdir) 

if __name__ == '__main__':
    
    os.system('clear');

    print('***********************************************');    
    print('***************** Clear all jobs ********');    
    print('***********************************************');     
    
    searchNClear('.')   
    
