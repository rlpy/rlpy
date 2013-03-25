######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################

# Run the main file multiple times and store the result of each run in a separate directory:
from main import *
from os import *

def unpackjob(args):
    main(*args)


StartID         = 1
FinishId        = 50
RUNS            = arange(StartID,FinishId+1)
PROJECT_PATH    = 'Results/TEST'
max_cpu         = multiprocessing.cpu_count()-2


#Create the ouput directory
checkNCreateDirectory(PROJECT_PATH)
#create the pool with the correct number of cpus
pool = Pool(max_cpu)
#pack the arguments into tuples
jobs = [(i, PROJECT_PATH, 1) for i in RUNS]
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
