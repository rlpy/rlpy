######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################

# Run the main file multiple times and store the result of each run in a separate directory:
import sys, os

RL_PYTHON_ROOT = '.'
while not os.path.exists(RL_PYTHON_ROOT+'/RL-Python/Tools'):
    RL_PYTHON_ROOT = RL_PYTHON_ROOT + '/..'
RL_PYTHON_ROOT += '/RL-Python'
RL_PYTHON_ROOT = os.path.abspath(RL_PYTHON_ROOT)
sys.path.insert(0, RL_PYTHON_ROOT)
from main import *

def unpackjob(args):
    main(*args)


StartID         = 1
FinishId        = 5
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
