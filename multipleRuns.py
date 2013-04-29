#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################

# Run the main file multiple times and store the result of each run in a separate directory:
import sys, os

from main import *

#RL_PYTHON_ROOT = '.'
#while not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
#    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
#RL_PYTHON_ROOT += '/RLPy'
#RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
#sys.path.insert(0, RL_PYTHON_ROOT)
def unpackjob(args):
    main(*args)

StartID         = 1
FinishId        = 30
RUNS            = arange(StartID,FinishId+1)
#PROJECT_PATH    = 'Results/TEST'
PROJECT_PATH    = '.'
max_cpu         = multiprocessing.cpu_count()/2-1

print "Found %d free CPUs (Not using HT)" % max_cpu
print "Running Jobs %d-%d..." % (StartID,FinishId)
#Create the ouput directory
#checkNCreateDirectory(PROJECT_PATH)
#create the pool with the correct number of cpus
#sys.exit()
pool = Pool(max_cpu)
#pack the arguments into tuples
jobs = [(i, PROJECT_PATH, 0) for i in RUNS]
pool.map(unpackjob, jobs)

#if(platform.system() == 'Windows'):
#    for i in RUNS: # .py files executable by default
#        print "Started job %d" % i
#        os.system('main.py %d %s %d %d &' % (i, PROJECT_PATH, 0, 1))
#else:
#    for i in RUNS:
#
#        os.system('chmod +x main.py')
#        os.system('./main.py %d %s %d %d &' % (i, PROJECT_PATH, 0, 1))
#        print "Started job %d" % i
