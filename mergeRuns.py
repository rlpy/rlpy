######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from Tools import *

#path = 'Results/Example_Project' 
#paths = ['Results/13ICML-BatchiFDD/PST/PST-iFDD-50000-65'] 
paths = ['Results/Temp/BlocksWorld-iFDD-100000-0.1-NoSparse'] 


colors      = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple']
styles      = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
MarkerSize  = 15
Legend      = False

merger = Merger(paths,colors = colors, styles= styles, markersize = MarkerSize, legend = Legend)
pl.ioff()
#print mergedData.means[0].shape
merger.plot('Return')
#merger.plot('Return','Time(s)')
#merger.plot('Steps')
#merger.plot('Steps','Learning Steps')
#merger.plot('Steps','Episodes')
merger.plot('Steps','Time(s)')
#merger.plot('Steps','Time(s)')
#merger.plot('Features','Episodes')
#merger.plot('Terminal')
pl.show()
