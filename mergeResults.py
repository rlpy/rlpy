######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from main import *
from os import *

class MergedData(Object):
    self.AXES = ['step','return','time','feature','traj_length','terminal']
    def __init__(self,paths):
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        means       = []
        std_errs    = [] 
        for p in paths:
            means, std_errs = parse(p)
            self.means.append(means)
            self.std_errs.append(std_errs)
    def plot(self,Y_axis,X_axis = 'step'):
        Y_axis = lower(Y_axis)
        if Y_axis in self.AXES:
            y_ind = self.AXES.index(Y_axis)
        else:
            print 'unknown Y_axis = %s', Y_axis
        if X_axis in self.AXES and X_axis in ['step','time']:
            x_ind = self.AXES.index(X_axis)
        else:
            print 'unknown X_axis = %s', X_axis
        for agent in range(agents):
            X   = self.means[x_ind]
            Y   = self.means[y_ind]
            Err = self.means[y_ind]
            pl.plot(X,Y, linewidth = 2)
            pl.fill_between(X, Y-Err, Y+Err)
        pl.show()
        self.save()
    def save(self):
        pl.save('')
        pl.
#result file has 6-by-n elements where each row is 
#0. Test Step
#1. Return
#2. Time(s)
#3. Features
#4. Traj-Steps
#5. Terminal



paths = {'Results/Test1', 
         'Results/Temp'}

mergedData = MergedData(paths)
mergedData.plot('performance')
mergedData.plot('features')
mergedData.plot('terminals')
mergedData.plot('steps')
mergedData.plot('time')
mergedData.plot('performance','time')
mergedData.plot('performance','step')


