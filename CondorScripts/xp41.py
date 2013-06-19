#!/usr/bin/env python
# Alborz Geramifard March 4th 2013 MIT
# This script prepares directories and main files for various parameters in one shot
# It also copies the multipleRuns.py
# Inputs:
# dictionary of parameters
# agent: agent name = {SARSA, Q-LEARNING, LSPI}
# a0: Boyan alpha decay parameters = {.1,1}
# N0: Boyan alpha decay parameters = {100,1000}
# iFDD-T: iFDD threshold
import os, sys, time, re, string
#find RL_ROOT and add it to path
from Script_Tools import *
RL_PYTHON_ROOT = findRLRoot()
sys.path.insert(0, RL_PYTHON_ROOT)

if __name__ == '__main__':
    
#    os.system('clear');
#
#    print('*********************************************************************');    
#    print('***************** Making Experiments on ACL Machine*********************');    
#    print('*********************************************************************');     
#    
#    cmd = 'makexp.py %s' % sys.args 
        # simply copy the main.py here
        os.system('rm -rf multipleRuns.py; cp %s/multipleRuns.py .' % RL_PYTHON_ROOT)
        print '>>>> Copied multipleRuns file here.'
    
        
