######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from Tools import *

path = 'Results/Example_Project' 

colors = ['r','b','g','k'] 
mergedData = MergedData(path,colors = colors)
#mergedData.plot('Return')
#mergedData.plot('Return','Time(s)')
#mergedData.plot('Steps')
#mergedData.plot('Steps','Time(s)')
mergedData.plot('Features')
#mergedData.plot('Terminal')
pl.ioff()
pl.show()

