######################################################
# Developed by Alborz Geramiard Nov 30th 2012 at MIT #
######################################################

# Run the main file multiple times and store the result of each run in a separate directory:
from main import *
from os import *
RUNS            = 10
PROJECT_PATH    = 'Results/Example_Project'
for i in arange(1,RUNS+1):
    os.system('chmod +x main.py')
    print "Started job %d" % i
    os.system('./main.py %d %s &' % (i, PROJECT_PATH))
