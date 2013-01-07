#!/usr/bin/env python
# See http://threebean.wordpress.com/2011/06/06/installing-from-pip-inside-python-or-a-simple-pip-api/

import sys
import pip.commands.install # for install_distributions below

FAILURE=0
SUCCESS=1

class InstallError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
# From wordpress source at top
def install_distributions(distributions):
    command = pip.commands.install.InstallCommand()
    opts, args = command.parser.parse_args()
    # TBD, why do we have to run the next part here twice before actual install
    requirement_set = command.run(opts, distributions)
    requirement_set = command.run(opts, distributions)
    requirement_set.install(opts)    

def isPackageInstalled(packageName):
    try:
        __import__(packageName) # __import__('X') is identical to import X
        return True
    except ImportError as err:
        return False

def installPackage(packageName):
    if(isPackageInstalled(packageName)):
        print packageName+' is already installed.'
    else:
        print packageName+' not yet installed: installing'
        install_distributions([packageName])
    
    # Now the package should be installed; test to be sure
    if(isPackageInstalled(packageName)):
        print packageName+' installation verified.'
    else:
        raise InstallError(packageName+' failed installation.')
    

def checkValidVersion():
    print "Your Python Version: "+sys.version
    if not((2,7) < sys.version_info < (3,0)):
        print "RL-Python requires 2.7 < version < 3.0, please re-install before proceeding."
        print "If you believe the correct version of Python is already installed,"
        print "ensure that you only have one active copy of python in your PATH environment variable."
        print "If you suspect competing installations of Python, see the FAQ section."
        print "http://stackoverflow.com/questions/7746240/in-bash-which-gives-an-incorrect-path-python-versions"
        raise InstallError("Incorrect Version Number")
    else:
        print "Python version valid."
    
if __name__ == '__main__':
    print """Welcome to the dependency installer for RL-Python.
Note that you may safely re-run this script at any time
(for example, if installation fails at an intermediate step.)
"""

    try:
        checkValidVersion()
                
        print "Beginning installation of required packages."
        installPackage('scipy')
        installPackage('numpy')
        installPackage('networkx')
        installPackage('matplotlib')
#        installPackage('graphviz') # Not a pip package
        installPackage('scikit-learn')
        
    except InstallError as err:
        print "Installer failed: "+err.value
        print "Please re-run after correcting the error above."
    finally: print "Dependency installer terminated"