######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from Tools import *

#path = 'Results/Example_Project' 
path = 'Results/13ICML-BatchiFDD' 

colors = ['r','b','g','k'] 
merger = Merger(path,colors = colors)
#print mergedData.means[0].shape
#merger.plot('Return')
#merger.plot('Return','Time(s)')
merger.plot('Steps')
#merger.plot('Steps','Time(s)')
#merger.plot('Features')
#merger.plot('Terminal')
pl.ioff()
pl.show()

