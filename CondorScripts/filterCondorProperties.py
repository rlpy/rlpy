#!/usr/bin/python
#
# Queries condor for status of all nodes, tokenizes each one, and allows
# for filtering by certain properties e.g. operating system.
#
# Provides the option to write a requirements file based on results.
#
# Set the dictionary filtered_terms in main() below.

import os, sys, time, re, string
from Script_Tools import *

# Contains all properties of a Condor Machine listed by condor status
class CondorMachine(object):
    # Note the capitalization below to agree with Condor definitions
    Name        = None
    OpSys       = None
    Arch        = None
    State       = None
    Activity    = None
    LoadAv      = None
    Mem         = None
    ActvtyTime  = None
    
    def __init__(self, name, opsys, arch, state, activity, loadav, mem, actvtytime):
        Name        = name
        Opsys       = opsys
        Arch        = arch
        State       = state
        Activity    = activity
        LoadAv      = loadav
        Mem         = mem
        ActvtyTime  = actvtytime
    
    def conditionsSatisfied(self, **kwargs):
        for paramName,paramValue in kwargs.iteritems():
            if not hasattr(self, paramName):
                print 'Condor Machines have no parameter %s, check spelling and try again' % paramName
                sys.exit(1)
            
            if(getattr(self, paramName) != paramValue): # This machine doesn't meet requirements
                return False
        return True # This machine matches all filter items
    
    def __str__(self):
        return string.join([getAttr(self,attr) for attr in dir(self)])
    
# Takes list of strings of the condor status format 
# eg slot2@cocosci-1.csail.mit.edu   LINUX   X86_64  Claimed Idle 0,000 1003 18+14:15:17
# And removes anything before the '@' symbol
def removeSlotFromNames(allLines):
    for ind,line in enumerate(allLines):
        slot_string = string.split(line, '@', 1) # Split on the '@' in at most 1 place
        allLines[ind] = slot_string[1] # Element 1 has @ portion removed
    return allLines

def getCondorMachines(uniqueLines):
    allMachines = []
    for line in uniqueLines:
        tL = string.split(line) # tL = tokenized Line
        if(len(tL) != 8):
            print 'Error in filterCondorProperties.py: the following line '
            print 'does not have the expected 8-token format:'
            print tL
            sys.exit(1)
        newMachine = CondorMachine(tL[0], tL[1], tL[2], tL[3], tL[4], tL[5], tL[6], tL[7])
        allMachines.append(newMachine)
    return allMachines

def filterCondorMachines(allMachines, filteredTerms):
    # Not sure how to use inbuilt filter() function with class methods
    return [machine for machine in allMachines if machine.conditionsSatisfied(**filteredTerms)] 

if __name__ == '__main__':
    CONDOR_STATUS_FILE = 'condorStatusFile.txt'
    FILTERED_TERMS = {'OpSys':'LINUX'} # See CondorMachine class for valid filter terms
    
    
    # Output condor status to file
    os.system(string.join(['condor status > ',CONDOR_STATUS_FILE]));
    
    # Get all lines from file, remove 'slot' preceding their names, remove duplicates
    allLines = getAllLines(CONDOR_STATUS_FILE)
    allLines = removeSlotFromNames(allLines)
    uniqueLines = getUniqueLines(allLines)
    
    # Obtain list of all 
    allMachines = getCondorMachines(uniqueLines)
    
    filteredMachines = filterCondorMachines(allMachines, FILTERED_TERMS)

    print ''
    print 'total number of machines: %d' % len(allMachines)
    print 'number of machines matching properties: %d' % len(filteredMachines)
    print 'Valid machines'
    for machine in filteredMachines:
        print machine
    
