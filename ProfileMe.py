from main import *
#from mainDP import *
import os
import cProfile
import pstats
import platform


Output = 'PST-1x10K-10Iterations-LSPI-CompactBinary-nonsparsesolver-nonsparseA.pdf'
print 'Profiling'
cProfile.run('main(-1,SHOW_FINAL_PLOT=0,PROJECT_PATH = "Results/Temp")', 'Profiling/profile.dat')
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