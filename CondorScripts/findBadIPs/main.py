#!/usr/bin/python
# See http://threebean.wordpress.com/2011/06/06/installing-from-pip-inside-python-or-a-simple-pip-api/

# This script identifies if the current machine it is running on is
# missing any required packages, and if so, adds the hostname to the text file 'bad_hosts.txt'.

import sys,os

#Add all paths
RL_PYTHON_ROOT = os.environ.get('RL_PYTHON_ROOT')
if (RL_PYTHON_ROOT == None):
    print 'Could not get environment variable RL_PYTHON_ROOT: \
    \nplease re-run installer script or see FAQ.txt. \nExiting.'
    sys.exit()

sys.path.insert(0, RL_PYTHON_ROOT)

from Script_Tools import *

crit_packages 		= ['numpy','scipy','sklearn']
opt_packages 		= []
#opt_packages 		= ['matplotlib']

BAD_HOST_FILE 		= 'bad_hosts.txt' # contains list of bad HOST's, without duplicates
GOOD_HOST_FILE		= 'good_hosts.txt'
MISSING_PKG_FILE	= 'missing_pkg_log.txt' # contains log of bad HOST's with their associated missing packages


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
	
