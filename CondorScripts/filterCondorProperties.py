#!/usr/bin/python
#
# Queries condor for status of all nodes, tokenizes each one, and allows
# for filtering by certain properties e.g. operating system.
#
# Provides the option to write a requirements file based on results.
#
# Set the dictionary filtered_terms in main() below.
#
# Note that condor_status -constraint provides similar functionality
# but doesn't work for some attributes like 'name', 'OpSys', etc.

import os, sys, time, re, string
from Script_Tools import *

# Contains all properties of a Condor Machine listed by condor status.
# see http://research.cs.wisc.edu/htcondor/manual/v7.6/condor_status.html
# for additional possible specifications
class CondorMachine(object):
    # Note the capitalization below to agree with Condor definitions
    Name        = None
    OpSys       = None
    Arch        = None
    State       = None
    Activity    = None
    LoadAvg     = None
    Memory      = None
#    ActvtyTime  = None
    
    def __init__(self, name, opsys, arch, state, activity, loadavg, memory):
        self.Name        = name
        self.OpSys       = opsys
        self.Arch        = arch
        self.State       = state
        self.Activity    = activity
        self.LoadAvg     = loadavg
        self.Memory      = memory
#        self.ActvtyTime  = actvtytime
    
    def conditionsSatisfied(self, **kwargs):
        for paramName,paramValue in kwargs.iteritems():
            if not hasattr(self, paramName):
                print 'Condor Machines have no parameter %s, check spelling and try again' % paramName
                sys.exit(1)
            
            if(getattr(self, paramName) != paramValue): # This machine doesn't meet requirements
                return False
        return True # This machine matches all filter items
    
    def __str__(self):
        return self.Name
        #return string.join([getattr(self,attr) for attr in dir(self)])
    
# Takes list of strings (lines of file) and only returns those which appear
# to be lines corresponding to status of a single condor machine
def removeNonMachineLines(allLines):
    return [line for line in allLines if isMachineLine(line)]

# Returns true if this line corresponds to a condor status line, false otherwise
def isMachineLine(line):
    tokenizedLine = string.split(line)
    if(len(tokenizedLine) != 7):
        print 'this is not a machine line: %s' % line
        return False # Expect 7 attributes
    elif(tokenizedLine[0] == 'Name' and tokenizedLine[1] == 'OpSys'):
        print 'this is a header line %s' % line
        return False # This is the header
    return True # Line appears to be machine status

# Takes list of strings of the condor status format 
# eg slot2@cocosci-1.csail.mit.edu   LINUX   X86_64  Claimed Idle 0,000 1003 18+14:15:17
# And removes anything before the '@' symbol
def removeSlotFromNames(allLines):
    for ind,line in enumerate(allLines):
        slot_string = string.split(line, '@', 1) # Split on the '@' in at most 1 place
        if(len(slot_string) > 1):
            allLines[ind] = slot_string[1] # Element 1 has @ portion removed
        else: # There was no @ symbol, ie no 'slots' on this machine, just use full name
            allLines[ind] = slot_string[0]
    return allLines

def getCondorMachines(uniqueLines):
    allMachines = []
    for line in uniqueLines:
        tL = string.split(line) # tL = tokenized Line
        if(len(tL) != 7):
            print 'Error in filterCondorProperties.py: the following line '
            print 'does not have the expected 7-token format:'
            print tL
            sys.exit(1)
        newMachine = CondorMachine(tL[0], tL[1], tL[2], tL[3], tL[4], tL[5], tL[6])
        allMachines.append(newMachine)
    return allMachines

def filterCondorMachines(allMachines, filteredTerms):
    # Not sure how to use inbuilt filter() function with class methods
    return [machine for machine in allMachines if machine.conditionsSatisfied(**filteredTerms)] 

if __name__ == '__main__':
    CONDOR_STATUS_FILE = 'condorStatusFile.txt'
    REQ_FILE = 'Requirements.txt'
    FILTERED_TERMS = {'OpSys':'LINUX'} # See CondorMachine class for valid filter terms
    # Must manually specify status below since condor automatically truncates otherwise.
    COMMAND = 'condor_status -format "%s " Name -format "%s " OpSys -format "%s " Arch -format "%s " State -format "%s " Activity -format "%s " LoadAvg -format "%s " Memory -format "\n" ArbitraryString'# If no '%' is specified, then string is printed regardless of the field name, thus "ArbitraryString" fieldname is given.
    
    # Output condor status to file
    os.system(string.join([COMMAND,' > ',CONDOR_STATUS_FILE]));
    
    # Get all lines from file, remove 'slot' preceding their names, remove duplicates
    allLines = getAllLines(CONDOR_STATUS_FILE)
    allLines = removeNonMachineLines(allLines)
    allLines = removeSlotFromNames(allLines)
    uniqueLines = getUniqueLines(allLines)
    # Obtain list of all 
    allMachines = getCondorMachines(uniqueLines)
    filteredMachines = filterCondorMachines(allMachines, FILTERED_TERMS)

    print ''
    print 'total number of machines: %d' % len(allMachines)
    print 'number of machines matching properties: %d' % len(filteredMachines)
    print ''
    print 'Now logging requirements file'
    requirementsFile= FileHelper(REQ_FILE)
    requirementsFile.open('w') # clear the contents
    for machine in filteredMachines:
        requirementsFile.log('Machine == \"%s\" && \\' % machine.Name)
    badMachinesFile.close()
    requirementsFile.close()
    sys.exit(0)
    
