#!/usr/bin/python
import os, sys, time, re, string, pprint, re, glob, socket

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
USERNAME            = 'rhklein'
FINALFILE           = 'results.txt'
GRAPH_DIR           = ' ~/Graphs/FeatureDiscovery/'
PYTHON_PATH          = ' /usr/bin/python'
FINISHED_RUNS_NUM   = 5   # Number of runs which is counted as a finished run for testing a learning rate

import os
def findRLRoot():
    RL_PYTHON_ROOT = '.'
    while not os.path.exists(RL_PYTHON_ROOT+'/RL-Python/Tools'):
        RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
    RL_PYTHON_ROOT += '/RL-Python'
    RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
    return RL_PYTHON_ROOT
def logKeyOLD(log):
    timeRE     = re.compile('R\[[0-9]*:[0-9]*:[0-9]*')
    temp1       = timeRE.search(log)
    if not temp1:
        return sys.maxint
    else:
        temp1 = temp1.group(0)[2:] 
        
    [h1,m1,s1] = re.split(':+',temp1)
    if len(h1.lstrip('0')) > 0: h1 = h1.lstrip('0') 
    if len(m1.lstrip('0')) > 0: m1 = m1.lstrip('0')
    if len(s1.lstrip('0')) > 0: s1 = s1.lstrip('0')
#    print h1, m1, s1
    try:
        h1 = eval(h1)    
        m1 = eval(m1)    
        s1 = eval(s1)    
    except:
        print h1,m1,s1
        print log
        s1 = m1 = h1 = 10000000
        
    return h1*10000+m1*100+s1
def logKey(log):
    if 'Return=' in log:
        value = log.split('Return=',1)[1]
        value = value.split(',',1)[0]
        value = eval(value)
        return -value
    else:
        return 1000000000
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

# BELOW is used for checking bad machines


class FileHelper(object):
	file	= None
	fileName= ''
    
	def __init__(self, fileName = None):
		self.fileName = fileName
	
	def open(self,readType):
		if not self.file:
			self.file = open(self.fileName,readType)
	
	def close(self):
		if self.file:
			self.file.close()
	
	# Returns a line without \n at end, or None if line is None.
	def myReadLine(self):
		if not self.file:
			print 'You need to open the file first! No action taken.'
		
		line = self.file.readline()
		if not line:
			return None
		if line.endswith("\n"):
			if(len(line) > 1):
				line = line[:-1] # Omit the \n
			else: return ' ' # This line is ONLY a "\n"
		return line
	   	
	def log(self,str):
    # Print something both in output and in a file
	   	print str
	   	self.file.write(str +'\n')
		
	def line(self):
		self.log(SEP_LINE)

def getUniqueLines(allLines):
	return set(allLines)	

def getAllLines(fileName):
    myFile = FileHelper(fileName)
    myFile.open('r') 
    allLines = []
    while True:
            line = myFile.myReadLine()
            if not line: break
            else: allLines.append(line)
    myFile.close()
    return allLines

def addText(fileName,text, fileOpFlag = 'a'): # default append text
	myFile = FileHelper(fileName)
	myFile.open(fileOpFlag)
	myFile.log(text)
	myFile.close()
		
def isPackageInstalled(packageName):
    try:
        __import__(packageName) # __import__('X') is identical to import X
        return True
    except ImportError as err:
        return False

# Returns ip address associated with machine script is running on
def getIPAddress():
	myIP = None
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("mit.edu",80)) # Arbitrary connection on port 80
	myIP = s.getsockname()[0]
	s.close()
	return myIP

def getHostName():
	return socket.gethostname()
    
