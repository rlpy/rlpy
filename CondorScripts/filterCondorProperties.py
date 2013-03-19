#!/usr/bin/python
#
# !!!!!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!!!!!!!
# Currently slots are not being removed from machine names, since on a given
# machine, different slots can have different properties
#
# Queries condor for specified attributes of all nodes, tokenizes each one, and allows
# for filtering by certain properties e.g. operating system.
#
# Provides the option to write a requirements file based on results.
#
#
# #### INSTRUCTIONS ####
#
# - Choose which attributes to output by adding them to the CondorMachine class below
# (don't forget about init()! ) (All other methods detect these new parameters and adjust automatically.)
# - Choose which attributes to filter by adding them to the dictionary 'filtered_terms' in main() below.
# - Note that in order to filter by some property, it must appear in the attributes
# of the CondorMachine class (ie can't filter by an attribute you didn't output)
#
# #######################
#
# Aside: condor_status -constraint provides similar functionality to script,
# but doesn't work for some attributes like 'Name', 'OpSys', etc.
#


import os, sys, time, re, string
from Script_Tools import *
from scipy.constants.constants import mach

# Contains all properties of a Condor Machine listed by condor status.
# see http://research.cs.wisc.edu/htcondor/manual/v7.6/condor_status.html
# for additional possible specifications
class CondorMachine(object):
    # Note the capitalization below to agree with Condor definitions
    Name        = None
    OpSys       = None
    Arch        = None
    Memory      = None
    KFlops      = None
#    ActvtyTime  = None

    NUM_ATTRIBUTES = None # assigned after class definition
    SORTED_ATTRIBUTES = None # assigned after class definition, sorted list of attributes
    
    def __init__(self, Name=None, OpSys=None, Arch=None, Memory=None, KFlops=None):
        self.Name        = Name
        self.OpSys       = OpSys
        self.Arch        = Arch
        self.Memory      = Memory
        self.KFlops      = KFlops
    
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
machineAttributes = vars(CondorMachine()) 
machineAttributes = machineAttributes.keys()
machineAttributes.sort()
CondorMachine.NUM_ATTRIBUTES = len(machineAttributes) # Initialize static variable
CondorMachine.SORTED_ATTRIBUTES = machineAttributes # Initialize static variable
del machineAttributes # finished initializing static variables, this is now an unused global variable, delete.

def generateCondorCommand():
    COMMAND = 'condor_status'
    for attr in CondorMachine.SORTED_ATTRIBUTES:
        COMMAND = COMMAND + ' -format "%s " ' + attr
    COMMAND = COMMAND + ' -format "\n" ArbitraryString' # If no '%' is specified, then string is printed regardless of the field name, thus "ArbitraryString" fieldname is given. 
    #COMMAND = 'condor_status -format "%s " Name -format "%s " OpSys -format "%s " Arch -format "%s " State -format "%s " Activity -format "%s " LoadAvg -format "%s " Memory -format "%s " KFlops -format "\n" ArbitraryString'# If no '%' is specified, then string is printed regardless of the field name, thus "ArbitraryString" fieldname is given. 
    return COMMAND
    
# Takes list of strings (lines of file) and only returns those which appear
# to be lines corresponding to status of a single condor machine
def removeNonMachineLines(allLines):
    return [line for line in allLines if isMachineLine(line)]

# Returns true if this line corresponds to a condor status line, false otherwise
def isMachineLine(line):
    tokenizedLine = string.split(line)
    if(len(tokenizedLine) != CondorMachine.NUM_ATTRIBUTES):
        print 'this is not a machine line: %s' % line
        return False # Expect NUM_ATTRIBUTES attributes
    # Below elif not needed with new '-format' specification for condor_status
#    elif(tokenizedLine[0] == 'Name' and tokenizedLine[1] == 'OpSys'):
#        print 'this is a header line %s' % line
#        return False # This is the header
    return True # Line appears to be machine status

# Takes list of strings of the condor status format 
# eg slot2@cocosci-1.csail.mit.edu   LINUX   X86_64  Claimed Idle 0,000 1003 18+14:15:17
# And removes anything before the '@' symbol
def removeSlotFromNames(allLines):
    for ind,line in enumerate(allLines):
        #slot_string = string.split(line, '@', 1) # Split on the '@' in at most 1 place
        #slot_string = string.split(line)
        
        #if(len(slot_string) > 1):
        #    allLines[ind] = slot_string[1] # Element 1 has @ portion removed
        #else: # There was no @ symbol, ie no 'slots' on this machine, just use full name
        #    allLines[ind] = slot_string[0]
        allLines[ind] = re.sub('slot.*@', '',line) # Looks for any instances of the text 'slot' and deletes up to the @ symbol.
    return allLines

# Requires that attributes on each line be sorted alphabetically, 
# since CondorMachine parameters are assigned under that assumption.
def getCondorMachines(sortedLines):
    allMachines = []
    for line in sortedLines:
        tL = string.split(line) # tL = tokenized Line
        if(len(tL) != CondorMachine.NUM_ATTRIBUTES):
            print 'Error in filterCondorProperties.py: the following line '
            print 'does not have the expected %d-token format:' % CondorMachine.NUM_ATTRIBUTES
            print tL
            sys.exit(1)
        # We assume that each line is sorted alphabetically by attribute, so we can match
        # each attribute to its parameter automatically:
        machineAttrAssignments = dict(zip(CondorMachine.SORTED_ATTRIBUTES, tL))
        # machineAttrAssignments is dictionary containing attr:value (eg 'Name':'cocsi.csail.mit.edu')
        newMachine = CondorMachine(**machineAttrAssignments)
        allMachines.append(newMachine)
    return allMachines

def getUniqueMachines(allMachines):
    foundNames = []
    uniqueMachines = []
    for machine in allMachines:
        if not machine.Name in foundNames:
            uniqueMachines.append(machine)
    return uniqueMachines

def filterCondorMachines(allMachines, filteredTerms):
    # Not sure how to use inbuilt filter() function with class methods
    return [machine for machine in allMachines if machine.conditionsSatisfied(**filteredTerms)] 

if __name__ == '__main__':
    CONDOR_STATUS_FILE = 'condorStatusFile.txt'
    REQ_FILE = 'Requirements.txt'
    FILTERED_TERMS = {'OpSys':'LINUX', 'Arch':'X86_64', 'KFlops':'1647069'} # See CondorMachine class for valid filter terms
    # Must specify status attributes in command below since condor automatically truncates otherwise.
    # We generate this command automatically, with ATTRIBUTES IN ALPHABETICAL ORDER
    # as required by later code.
    COMMAND = generateCondorCommand()
    
    # Output condor status to file
    os.system(string.join([COMMAND,' > ',CONDOR_STATUS_FILE]));
    
    # Get all lines from file, remove 'slot' preceding their names, remove duplicates
    allLines = getAllLines(CONDOR_STATUS_FILE)
    allLines = removeNonMachineLines(allLines)
    # CURRENTLY WE ARE RETAINING SLOTS IN NAMES because on a single machine,
    # different slots have different values for memory, etc.
    #allLines = removeSlotFromNames(allLines)
    # Obtain list of all 
    allMachines = getCondorMachines(allLines)
    uniqueMachines = getUniqueMachines(allMachines)
    filteredMachines = filterCondorMachines(allMachines, FILTERED_TERMS)

    print 'Now logging requirements file'
    requirementsFile= FileHelper(REQ_FILE)
    requirementsFile.open('w') # clear the contents
    for machine in filteredMachines:
        requirementsFile.log('Machine == \"%s\" || \\' % machine.Name)
    requirementsFile.close()

    print ''
    print 'total number of machines: %d' % len(allMachines)
    print 'number of machines matching properties: %d' % len(filteredMachines)
    print ''

    sys.exit(0)
    
