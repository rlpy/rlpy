#!/usr/bin/env python
# See http://threebean.wordpress.com/2011/06/06/installing-from-pip-inside-python-or-a-simple-pip-api/

import sys

class ExitCode:
	NONE_MISSING =0
	NUMPY_MISSING=1
	SCIPY_MISSING=2
	BOTH_MISSING = NUMPY_MISSING + SCIPY_MISSING # optional

def isPackageInstalled(packageName):
    try:
        __import__(packageName) # __import__('X') is identical to import X
        return True
    except ImportError as err:
        return False
    
if __name__ == '__main__':
    returnStatus = ExitCode.NONE_MISSING
    if not isPackageInstalled('numpy'):
   		returnStatus = returnStatus + ExitCode.NUMPY_MISSING
    if not isPackageInstalled('scipy'):
    	returnStatus = returnStatus + ExitCode.SCIPY_MISSING
       
    print returnStatus
    sys.exit(returnStatus)