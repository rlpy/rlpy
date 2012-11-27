from main import *
import os
import cProfile
print 'Profiling'
#cProfile.run('main(1,0)', 'Profiling/profile.dat')
command = './Profiling/gprof2dot.py -f pstats ./Profiling/profile.dat > ./Profiling/graph.txt'
os.system(command)
command = '/usr/local/bin/dot -T pdf ./Profiling/graph.txt -o ./Profiling/graph.pdf'
os.system(command)
print 'Done!'