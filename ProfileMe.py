from main import *
import os
import cProfile
import pstats


Output = 'SARSA-Chain10-10K-iTabular'
print 'Profiling'
#                          cProfile.run('main(1,0)', 'Profiling/profile.dat')
p = pstats.Stats('Profiling/profile.dat')
p.sort_stats('time').print_stats(5)

#Make sure gprof2dot is executable
command = 'chmod +x Profiling/gprof2dot.py'
os.system(command)

#Load the STATS and prepare the dot file for graphvis 
command = './Profiling/gprof2dot.py -f pstats ./Profiling/profile.dat > ./Profiling/graph.txt'
os.system(command)

#Call Graphvis to generate the pdf 
command = '/usr/local/bin/dot -T pdf ./Profiling/graph.txt -o ./Profiling/'+Output+'.pdf'
os.system(command)
print 'Done!'