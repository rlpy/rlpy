#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from main import *
#from mainDP import *
import os
import cProfile
import pstats
import platform


Output = 'AFTER-CANOPY.pdf'
print 'Profiling'
cProfile.run('main(-1,PROJECT_PATH = "Results/Temp")', 'Profiling/profile.dat')
p = pstats.Stats('Profiling/profile.dat')
p.sort_stats('time').print_stats(5)

if(platform.system() != 'Windows'):
    #Make sure gprof2dot is executable
    command = 'chmod +x Profiling/gprof2dot.py'
    os.system(command)

if(platform.system() == 'Windows'):
    #Load the STATS and prepare the dot file for graphvis 
    command = '.\Profiling\gprof2dot.py -f pstats .\Profiling\profile.dat > .\Profiling\graph.txt'
    os.system(command)
    
        #Call Graphvis to generate the pdf 
    command = 'dot -T pdf .\Profiling\graph.txt -o .\Profiling\\'+Output
    os.system(command)
    
    
    
else:
    #Load the STATS and prepare the dot file for graphvis 
    command = './Profiling/gprof2dot.py -f pstats ./Profiling/profile.dat > ./Profiling/graph.txt'
    os.system(command)
    
    #Call Graphvis to generate the pdf 
    command = '/usr/local/bin/dot -T pdf ./Profiling/graph.txt -o ./Profiling/'+Output
    os.system(command)


print 'Done!'