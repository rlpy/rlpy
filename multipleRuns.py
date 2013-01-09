######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################

# Run the main file multiple times and store the result of each run in a separate directory:
from main import *
from os import *

StartID         = 1
FinishId        = 5
RUNS            = arange(StartID,FinishId+1)
#PROJECT_PATH    = 'Results/13ICML-BatchiFDD'
PROJECT_PATH    = 'Results/13ICML-SARSA'
max_cpu         = multiprocessing.cpu_count()
for i in RUNS:
    os.system('chmod +x main.py')
    os.system('./main.py %d %s &' % (i, PROJECT_PATH))
    print "Started job %d" % i
