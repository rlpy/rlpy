#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Bob Klein
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        print "RLPy requires 2.7 < version < 3.0, please re-install before proceeding."
        print "If you believe the correct version of Python is already installed,"
        print "ensure that you only have one active copy of python in your PATH environment variable."
        print "If you suspect competing installations of Python, see the FAQ section."
        print "http://stackoverflow.com/questions/7746240/in-bash-which-gives-an-incorrect-path-python-versions"
        raise InstallError("Incorrect Version Number")
    else:
        print "Python version valid."
    
if __name__ == '__main__':
    print """Welcome to the dependency installer for RLPy.
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