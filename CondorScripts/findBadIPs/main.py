#!/usr/bin/python
# See http://threebean.wordpress.com/2011/06/06/installing-from-pip-inside-python-or-a-simple-pip-api/

import sys, socket
from sets import Set

crit_packages 		= ['numpy','scipy','sklearn']
opt_packages 		= ['matplotlib']

BAD_HOST_FILE 		= 'bad_hosts.txt' # contains list of bad HOST's, without duplicates
GOOD_HOST_FILE		= 'good_hosts.txt'
MISSING_PKG_FILE	= 'missing_pkg_log.txt' # contains log of bad HOST's with their associated missing packages

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
			line = line[:-1] # Omit the \n
		return line
	   	
	def log(self,str):
    # Print something both in output and in a file
	   	print str
	   	self.file.write(str +'\n')
		
	def line(self):
		self.log(SEP_LINE)

def getUniqueLines(fileName):
	myFile = FileHelper(fileName)
	myFile.open('r')
	allLines = [myFile.myReadLine() ]
	myFile.close()
	return Set(allLines)	

def addText(fileName,text):
	myFile = FileHelper(fileName)
	myFile.open('a')
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
    
if __name__ == '__main__':
	missingPkgFile = FileHelper(MISSING_PKG_FILE)
	missingPkgFile.open('a')
#	myIP = getIPAddress()
	myHostName = getHostName()
	isBadMachine = False
	for critPkg in crit_packages:
		if not isPackageInstalled(critPkg):
			missingPkgFile.log(myHostName+':\t\t MISSING CRITICAL PACKAGE:\t'+critPkg)
			isBadMachine = True
	
	for optPkg in opt_packages:
		if not isPackageInstalled(optPkg):
			missingPkgFile.log(myHostName+':\t\t missing optional package:\t'+optPkg)
    
	if isBadMachine: # This computer sucks.
		bad_machines = getUniqueLines(BAD_HOST_FILE)
		if not myHostName in bad_machines: # This is the first time we've encountered this bad computer.
			addText(BAD_HOST_FILE,myHostName)
			missingPkgFile.log('Computer '+myHostName+' NEWLY added as FAULTY.\n')
		else:
			missingPkgFile.log('Computer '+myHostName+' Faulty, but already discovered.\n')
	else:
		addText(GOOD_HOST_FILE,myHostName)
	missingPkgFile.close()
	sys.exit(0)
	