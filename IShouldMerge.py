#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot
# Copy this file in the parent directory where the runs of all of your algorithms reside.
# e.g:
# /Project
# /Project/Algorithm1
# /Project/Algorithm2
# /Project/Algorithm3
# /Project/Algorithm4
# copy merge.py in /Project
# OR set the path to the correct location
# If you want to select a set of runs you can pass a list of paths that you would like to be considered.

#!/usr/bin/env python
######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

#Locate RLPy
#================
import sys, os
RL_PYTHON_ROOT = '.'
while os.path.abspath(RL_PYTHON_ROOT) != os.path.abspath(RL_PYTHON_ROOT + '/..') and not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
if not os.path.exists(RL_PYTHON_ROOT+'/RLPy/Tools'):
    print 'Error: Could not locate RLPy directory.' 
    print 'Please make sure the package directory is named RLPy.'
    print 'If the problem persists, please download the package from http://acl.mit.edu/RLPy and reinstall.'
    sys.exit(1)
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT + '/RLPy')
sys.path.insert(0, RL_PYTHON_ROOT)

from Tools import *

#paths = ['.']
paths = ['Results/I-Should-Run']
#paths = ['Results/IShouldRun/Pendulum_InvertedBalance-Tabular-20000']

labels              = []
colors              = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles              = ['o', 'v', '8', 's', 'p', '*', '<','h', '^', 'H', 'D',  '>', 'd']
MarkerSize          = 7    
Legend              = True  # Draw legends?
maxSamples          = inf   # Maximum number of samples to be loaded from the directory. If set inf it will use all of them
minSamples          = 1     # Minimum number of samples required to include a directory for plotting 

merger = Merger(paths,labels=labels, colors = colors, styles= styles, markersize = MarkerSize, legend = Legend, maxSamples = maxSamples, minSamples = minSamples)
pl.ioff()

#print mergedData.means[0].shape
#merger.plot('Return')
#merger.plot('Return','Features')
merger.plot('Return','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Steps','Features')
#merger.plot('Steps')
merger.plot('Steps','Learning Steps')
#merger.plot('Steps','Episodes')
#merger.plot('Steps','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Features','Time(s)')
#merger.plot('Terminal')
# FOR POLICY EVALUTION
#######################
#merger.plot("Error",'Iterations')
#merger.plot("Error",'Features')
#merger.plot("Error",'Time(s)')
 
merger.plot() #Works both with control and PE plotting the most common output

pl.show()


