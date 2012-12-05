######################################################
# Developed by Alborz Geramiard Dec 2nd 2012 at MIT #
######################################################
# Merge multiple results of several algorithms and show them on one plot

from main import *
from os import *
import glob

class MergedData(object):
    AXES = ['Learning Steps','Return','Time(s)','Features','Steps','Terminal']
    def __init__(self,paths,output_path,colors,bars=1):
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        self.means       = []
        self.std_errs    = [] 
        self.bars   = bars  #Draw bars?
        self.colors = colors
        self.output_path = output_path
        self.experiments = len(paths) 
        self.means  = []
        self.std_errs = [] 
        self.fig = pl.figure(1)
        self.datapoints_per_graph = None # Number of datapoints to be shown for each graph (often this value is 10 corresponding to 10 performance checks)
        for p in paths:
            means, std_errs = self.parse(p)
            self.means.append(means)
            self.std_errs.append(std_errs)
    def parse(self,path):
        # Parses all the files in form of <number>-results.txt and return 
        # two matricies corresponding to mean and std_err
        files       = glob.glob('%s/*-result.txt'%path)
        samples_num = len(files)
        if samples_num == 0:
            print 'Error: %s is empty!' % path
        print "%s => %d Samples" % (path,samples_num)
        #read the first file to initialize the matricies
        rows, cols  = loadtxt(files[0]).shape
        samples     = zeros((rows,cols,samples_num)) 
        for i,f in enumerate(files):
            samples[:,:,i] = loadtxt(files[i])
        _,self.datapoints_per_graph,_ = samples.shape
        return mean(samples,axis=2),std(samples,axis=2)/sqrt(samples_num)
    def plot(self,Y_axis,X_axis = 'Learning Steps'):
        self.fig.clear()
        min_ = +inf
        max_ = -inf    
        if Y_axis in self.AXES:
            y_ind = self.AXES.index(Y_axis)
        else:
            print 'unknown Y_axis = %s', Y_axis
        if X_axis in [self.AXES[0],self.AXES[2]]:
            x_ind = self.AXES.index(X_axis)
        else:
            print 'unknown X_axis = %s', X_axis
        Xs      = zeros((self.experiments,self.datapoints_per_graph))
        Ys      = zeros((self.experiments,self.datapoints_per_graph))
        Errs    = zeros((self.experiments,self.datapoints_per_graph))
        for i in range(self.experiments):
            X   = self.means[i][x_ind,:]
            Y   = self.means[i][y_ind,:]
            Err = self.std_errs[i][y_ind,:]
            pl.plot(X,Y,'-o', linewidth = 2,alpha=.5,color = self.colors[i],)
            if self.bars:
                pl.fill_between(X, Y-Err, Y+Err,alpha=.1, color = self.colors[i])
                max_ = max(max(Y+Err),max_); min_ = min(min(Y-Err),min_)
            else:
                max_ = max(Y.max(),max_); min_ = min(Y.min(),min_)
            Xs[i,:]     = X
            Ys[i,:]     = Y
            Errs[i,:]   = Err
        pl.xlim(0,max(Xs[:,-1]))
        pl.ylim(min_-.1*abs(max_-min_),max_+.1*abs(max_-min_))
        pl.xlabel(X_axis,fontsize=16)
        pl.ylabel(Y_axis,fontsize=16)
        pl.show()
        self.save(Y_axis,X_axis,Xs,Ys,Errs)
    def save(self,Y_axis,X_axis,Xs,Ys,Errs):
        fullfilename = self.output_path + '/' +Y_axis+'-by-'+X_axis
        checkDirectory(fullfilename)
        self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1)
        finalArray = vstack((Xs,Ys,Errs))
        savetxt(fullfilename+'.txt',finalArray, fmt='%0.4f', delimiter='\t')
        print "Saved the plot in both txt and pdf formats => %s" % fullfilename

paths = ['Results/BlocksWorld-SARSA-iFDD', 
         'Results/BlocksWorld-SARSA-IndependentDiscritization']
#paths = ['Results/Test1'] 

colors = ['r','b','g','k'] 
mergedData = MergedData(paths,'Results',colors = colors)
#mergedData.plot('Return')
#mergedData.plot('Return','Time(s)')
#mergedData.plot('Steps')
mergedData.plot('Features')
#mergedData.plot('Terminal')
pl.ioff()
pl.show()

