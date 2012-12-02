######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from main import *
from os import *

class MergedData(Object):
    self.AXES = ['step','return','time','feature','traj_length','terminal']
    def __init__(self,paths,bars = 0):
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        means       = []
        std_errs    = [] 
        self.bars   = bars  #Draw bars?
        self.colors = colors
        for p in paths:
            means, std_errs = parse(p)
            self.means.append(means)
            self.std_errs.append(std_errs)
    def plot(self,Y_axis,X_axis = 'step'):
        Y_axis = lower(Y_axis)
        min_ = inf
        max_ = +inf    
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
            pl.plot(X,Y, linewidth = 2,color = self.colors[agent])
            if self.bars:
                pl.fill_between(X, Y-Err, Y+Err,alpha=.1, color = self.colors[agent])
                max_ = max(Y+Err,max_); min_ = min(Y-Err,min_)
            else:
                 max_ = max(Y,max_); min_ = min(Y,min_)
        pl.xlim(0,X[-1])
        pl.ylim(min_-.1*abs(min_),max_+.1*abs(max_))
        pl.show()
        self.save()
    def save(self):
        pl.xlabel('steps',fontsize=16)
        pl.ylabel('Performance',fontsize=16)
        performance_fig.savefig(self.fullpath+'/performance.pdf', transparent=True, pad_inches=.1)
#result file has 6-by-n elements where each row is 
#0. Test Step
#1. Return
#2. Time(s)
#3. Features
#4. Traj-Steps
#5. Terminal



paths = ['Results/Test1', 
         'Results/Temp']

colors = ['r','b','g','k'] 
mergedData = MergedData(paths,bars = 1,colors = colors)
mergedData.plot('performance')
mergedData.plot('features')
mergedData.plot('terminals')
mergedData.plot('steps')
mergedData.plot('time')
mergedData.plot('performance','time')
mergedData.plot('performance','step')


