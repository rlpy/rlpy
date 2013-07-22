from GeneralTools import *

class Merger(object):
    CONTROL_AXES    = ['Learning Steps','Return','Time(s)','Features','Steps','Terminal','Episodes'] #,'Discounted Return']
    PE_AXES         = ['Iterations','Features','Error','Time(s)']
    MDPSOLVER_AXES  = ['Bellman Updates', 'Return', 'Time(s)', 'Features', 'Steps', 'Terminal', 'Discounted Return', 'Iterations','Iteration Time']
#     DYNMEANS_AXES   = ['Outer Iterations', 'Steps','Features', 'MSE (Observed)', 'MSE','Time(s)', 'Number of Clusters','Objective']
    DYNMEANS_AXES   = ['Outer Iterations', 'Steps','Features', 'MSE', 'Time(s)', 'Number of Clusters','Objective','Cluster Accuracy', 'Batch Number']

    
    prettyText = 1 #Use only if you want to copy paste from .txt files otherwise leave it to 0 so numpy can read such files.
    def __init__(self,paths, labels = None, output_path = None,
                 colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','purple'],
                 styles = ['o', 'v', '8', 's', 'p', '*', '<','h', '^', 'H', 'D',  '>', 'd'],
                 markersize = 5, xbars=1, ybars=1, legend = False, maxSamples = inf, minSamples = 1,
                 getMAX = 0,showSplash=True, numLabelSuffixes=2, token_split_char='-'):
             
        #import the data from each path. Results in each of the paths has to be consistent in terms of size
        self.means                  = []
        self.std_errs               = []
        self.ybars                  = ybars  #Draw error bars?
        self.xbars                  = xbars
        self.xbar_interval          = 1     # Only plot x error bars in this interval (eg every 5 datapoints)
        self.colors                 = colors
        self.styles                 = styles
        self.markersize             = markersize
        self.legend                 = legend
        self.maxSamples             = maxSamples # In case we want to use less than available number of samples
        self.minSamples             = minSamples # Directories with samples less than this value will be ignored.
        self.samples                = []         # Number of Samples for each experiments
        self.useLastDataPoint       = False     # By default assume all data has the same number of points along the X axis.
        self.getMAX                 = getMAX    # Instead of mean of all experiments find the best performance among them
        self.output_path            = output_path
        self.labels                 = labels
        self.showSplash             = showSplash  # No figures just txt output
        self.num_label_suffixes     = numLabelSuffixes # Number of string tokens to include between token_split_char characters (eg '_') when producing legend
        self.token_split_char       = token_split_char # Character to use when splitting string token names
      
        self.legend_handles         = []    # Selectively add plotted items to legend
        
        #Extract experiment paths by finding all subdirectories in all given paths that contain experiment.
        self.extractExperimentPaths(paths)
        #Setup the output path if it has not been defined:
        if output_path == None:
            self.output_path = '.' if len(paths) > 1 else paths[0]

        #Setup Labels if not defined
        if labels == None or labels == []: self.labels = [self.extractLabel(p, self.token_split_char) for p in self.exp_paths]
        #print "Experiment Labels: ", self.labels

        self.exp_num                = len(self.exp_paths)
        self.means                  = [] # Matrix with rows = metrics, cols = datapoints
        self.std_errs               = []
        self.time_95_mean           = []
        self.time_95_std_errs       = []
        if not isOnCluster() and self.showSplash:
            self.fig                    = pl.figure(1)
        self.datapoints_per_graph   = [] # Number of datapoints to be shown for each graph (often this value is 10 corresponding to 10 performance checks)
        
        
        if len(self.exp_paths) == 0:
            print "No directory found with at least %d result files at %s." % (self.minSamples, paths)
            return
        for expInd, exp in enumerate(self.exp_paths):
            means, std_errs, samples, time_95_mean, time_95_std_errs,dataPoints = self.parseExperiment(exp)
            self.means.append(means)
            self.std_errs.append(std_errs)
            self.samples.append(samples)
            self.time_95_mean.append(time_95_mean)
            self.time_95_std_errs.append(time_95_std_errs)
            self.datapoints_per_graph.append(dataPoints)
    def extractExperimentPaths(self,paths):
        self.exp_paths = []
        for path in paths:
            for p in os.walk(path):
                dirname = p[0]
                if self.hasResults(dirname):
                   self.exp_paths.append(dirname)
        if self.labels == None:
            self.exp_paths  = sorted(self.exp_paths)
    def extractLabel(self,p, token_split_char):
        # Given an experiment directoryname it extracts a reasonable label
        tokens = p.split(token_split_char)
        firstTokenInd = len(tokens) - self.num_label_suffixes
        if firstTokenInd < 0: print 'ERROR: attempted to print more experiment labels than exist.'
        expName = ''
        for ind in range(firstTokenInd, len(tokens)-1):
            expName += tokens[ind] + '-'
        expName += tokens[-1]
        return expName
#         if len(tokens) > 1:
#             return tokens[-self.num_label_suffixes]+'-'+tokens[-1]
#         else:
#             return tokens[-1]
    def parseExperiment(self,path):
        # Parses all the files in form of <number>-results.txt and return
        # two matrices corresponding to mean and std_err
        # If the number of points along the x axis is not identical across all runs and experiments it returns the last data point for all experiments
        # This function also keeps track of the time it took each method to reach its 95% of its final performance without dropping afterwards.
        if '/' in path: 
            _,_,fringeDirName = path.rpartition('/') 
        else:
             fringeDirName = path
             
        files       = glob.glob('%s/*-results.txt'%path)
        samples_num = len(files)
        if self.maxSamples < samples_num:
            print "%d/%d Samples <= %s" % (samples_num, self.maxSamples,fringeDirName)
            samples_num = self.maxSamples
        else:
            print "%d Samples <= %s" % (samples_num, fringeDirName)
            
        # Loop over all files to find max number of columns = x-axis datapoints (performance checks)
        maxCols = 0
        for file in files:
            tmpMatr = readMatrixFromFile(file)
            maxCols = max(maxCols, tmpMatr.shape[1])
            
        # Determine number of rows, used to determine experiment type
        matrix      = readMatrixFromFile(files[0])
        num_metrics, _  = matrix.shape
        
        if num_metrics == len(self.DYNMEANS_AXES):
            self.ResultType = 'Dynamic Means'
            self.AXES = self.DYNMEANS_AXES
        elif num_metrics == len(self.PE_AXES):
            self.ResultType = 'Policy Evaluation'
            self.AXES = self.PE_AXES
        elif num_metrics == len(self.CONTROL_AXES):
            self.ResultType = 'RL-Control'
            self.AXES = self.CONTROL_AXES
        else:
            self.ResultType = 'MDPSolver'
            self.AXES = self.MDPSOLVER_AXES

        # Allocate mask for results (padding mask with ones where samples do not exist)
        # Third dimension is # of results.txt files in experiment directory
        mask = ones((num_metrics, maxCols, samples_num))
            
        #Extract time to get to 95% of performance
        
        dataLengths = zeros(samples_num)
        
        if not self.useLastDataPoint:
            samples     = zeros((num_metrics,maxCols,samples_num))
            times_95    = zeros(samples_num)

            for resultInd,f in enumerate(files):
                if resultInd == samples_num: break
                M = readMatrixFromFile(files[resultInd])
                
                _, tmpCols = M.shape
                dataLengths[resultInd] = tmpCols
                #print M.shape
                try:
                    samples[:,:tmpCols,resultInd] = M
                    mask[:,:tmpCols,resultInd] = 0
                    if('Return') in self.AXES:
                        times_95[resultInd] = self.extract_time_95(M)

                except:
                    print "Unmatched number of points along X axis: %d != %d" % (maxCols,M.shape[1])
                    self.useLastDataPoint = True
                    break
        if self.useLastDataPoint:
            print 'WARNING: self.useLastDataPoint not yet supported'
            # TODO FIXME - incorporate mask above
            samples     = zeros((num_metrics,1,samples_num))
            times_95    = zeros(samples_num)
            for i,f in enumerate(files):
                if i == samples_num: break
                M = readMatrixFromFile(files[i])
                if('Return') in self.AXES:
                    times_95[i] = self.extract_time_95(M)
                else:
                    print 'Could not extract time to 95%% performance:'
                    print 'No \'Return\' field in axes: ', self.AXES
                samples[:,:,i] = M[:,-1].reshape((-1,1)) # Get the last column of the matrix

        
        masked_samples = ma.masked_array(samples, mask=mask)

        
#         print '#### std ', masked_samples.std(axis=2)
#         print '#### sqrts ', sqrt(dataLengths)
#         print '#### newvals ', masked_samples.std(axis=2) / sqrt(dataLengths)
     
        # Give stderr instead of std deviation by using masked stats (mstats) method "sem"
        # Note that the stats method stats.sem ignores the mask, need stats.mstats.
        if self.getMAX:
#             return ma.max(masked_samples,axis=2),masked_samples.std(axis=2), samples_num, mean(times_95), std(times_95)/sqrt(samples_num), maxCols, batchMeans, batchStds
            return ma.max(masked_samples,axis=2),stats.mstats.sem(masked_samples, axis=2), samples_num, mean(times_95), std(times_95)/sqrt(samples_num), maxCols

        else:
            meanVals = masked_samples.mean(axis=2)
            stdErrVals = stats.mstats.sem(masked_samples, axis=2)
            
            # Anywhere have masked value, unmask and set zero.
            stdErrVals[stdErrVals.mask==True] = 0
            stdErrVals.mask[stdErrVals.mask==True] = False
#            print 'means,stdd',meanVals,stdErrVals
            
#             return masked_samples.mean(axis=2),masked_samples.std(axis=2), samples_num, mean(times_95), std(times_95)/sqrt(samples_num), maxCols, batchMeans, batchStds
            return meanVals, stdErrVals, samples_num, mean(times_95), std(times_95)/sqrt(samples_num), maxCols

    def showLast(self,Y_axis = None):
        # Prints the last performance of all experiments
        if Y_axis == None:
            if self.ResultType == 'Policy Evaluation': Y_axis = 'Error'
            elif self.ResultType == 'Dynamic Means': Y_axis = 'L2 Observed Transition Errors'
            else: Y_axis = 'Return'
        y_ind = self.AXES.index(Y_axis)
        M = array([M[y_ind,-1] for M in self.means])
        V = array([V[y_ind,-1] for V in self.std_errs])
        #os.system('clear');
        print "=========================="
        print "Last Performance+std_errs"
        print "=========================="
        for i,e in enumerate(self.exp_paths):
            print "%s:\t%0.3f+%0.3f based on %d Samples" % (e,M[i],V[i],self.samples[i])
    def bestExperiment(self,Y_axis = None, mode = 0):
        # Returns the experiment with the best final results
        # Best is defined in two settings:
        # Mode 0: Final Mean+variance
        # Mode 1: Area under the curve
        if Y_axis == None:
            if self.ResultType == 'Policy Evaluation': Y_axis = 'Error'
            elif self.ResultType == 'Dynamic Means': Y_axis = 'L2 Observed Transition Errors'
            else: Y_axis = 'Return'
        y_ind = self.AXES.index(Y_axis)
        if mode == 0:
            M = array([M[y_ind,-1] for M in self.means])
            V = array([V[y_ind,-1] for V in self.std_errs])
            best_index = argmax(M+V)
            best = max(M+V)
        else:
            M = array([M[y_ind,:] for M in self.means])
            S = sum(M,axis=1)
            print M
            print S
            best = max(S)
            best_index = argmax(S)
        return self.exp_paths[best_index], best
    def plot(self,Y_axis = None, X_axis = None):
        #Setting default values based on the Policy Evaluation or control
        if Y_axis == None:
            if self.ResultType == 'Policy Evaluation': Y_axis = 'Error'
            elif self.ResultType == 'Dynamic Means': Y_axis = 'L2 Observed Transition Errors'
            else: Y_axis = 'Return'
        if X_axis == None:
            if self.ResultType == 'RL-Control':
                X_axis = 'Learning Steps'
            elif self.ResultType == 'MDPSolver':
                X_axis = 'Time(s)'
            elif self.ResultType == 'Dynamic Means': X_axis = 'Outer Iterations'
            else:
                X_axis = 'Iterations'

        if not isOnCluster() and self.showSplash: self.fig.clear()
        min_ = +inf
        max_ = -inf

        #See if requested axes are reasonable
        if Y_axis in self.AXES:
            y_ind = self.AXES.index(Y_axis)
        else:
            print 'unknown Y_axis = %s', Y_axis
            print 'Allowed values:'
            print self.AXES
        if X_axis in self.AXES:
            x_ind = self.AXES.index(X_axis)
        else:
            print 'unknown X_axis = %s', X_axis
            print 'Allowed values:'
            print self.AXES

        Xs      = zeros((self.exp_num,max(self.datapoints_per_graph)))
        Ys      = zeros((self.exp_num,max(self.datapoints_per_graph)))
        X_Errs  = zeros((self.exp_num,max(self.datapoints_per_graph)))
        Y_Errs  = zeros((self.exp_num,max(self.datapoints_per_graph)))

        for i in arange(self.exp_num):
            X   = self.means[i][x_ind,:] # Look at parameters setting i, metric x_ind, for all datapoints
            Y   = self.means[i][y_ind,:]
            X_Err = self.std_errs[i][x_ind,:] # For x-error bars, if desired
            Y_Err = self.std_errs[i][y_ind,:]

#            X = masked_X
#            Y = masked_Y
#            X_Err = masked_X_Err
#            Y_Err = masked_Y_Err
                

            if not isOnCluster() and self.showSplash:
                if len(X) == 1 and self.ybars:
                    if self.xbars:
                        pl.errorbar(X, Y, xerr=X_Err, yerr=Y_Err, marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7,color = self.colors[i%len(self.colors)],markersize = self.markersize, label = self.labels[i])
                    else: pl.errorbar(X, Y, yerr=Y_Err, marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7,color = self.colors[i%len(self.colors)],markersize = self.markersize, label = self.labels[i])
                    max_ = max(max(Y+Y_Err),max_); min_ = min(min(Y-Y_Err),min_)
                else:
                    if self.xbars:
#                         pl.errorbar(X, Y, xerr=X_Err, marker = self.styles[i%len(self.styles)], linewidth = 1,alpha=.7, color=self.colors[i%len(self.colors)], ecolor='k',
#                                     markersize = self.markersize, label = self.labels[i])
                        numBarsToPlot = ceil(len(X_Err) / self.xbar_interval)
                        if self.xbar_interval > 1:
                            for j in arange(numBarsToPlot):
                                plottedInd = j*self.xbar_interval
                                X_Err[plottedInd+1:plottedInd+self.xbar_interval] = zeros(self.xbar_interval-1)
                        # Else we're plotting all error bars, make no changes
                        pl.errorbar(X, Y, xerr=X_Err, linestyle ='-', marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7, color=self.colors[i%len(self.colors)], ecolor='k',
                                    markersize = self.markersize, label = self.labels[i])
                    else:
                        h = pl.plot(X,Y,linestyle ='-', marker = self.styles[i%len(self.styles)], linewidth = 2,alpha=.7,color = self.colors[i%len(self.colors)],markersize = self.markersize, label = self.labels[i])
                        self.legend_handles.append(h) # add to legend
                    if self.ybars:
                        pl.fill_between(X, Y-Y_Err, Y+Y_Err,alpha=.3, color = self.colors[i%len(self.colors)])
                        max_ = max(max(Y+Y_Err),max_); min_ = min(min(Y-Y_Err),min_)
                    else:
                        max_ = max(Y.max(),max_); min_ = min(Y.min(),min_)
        
#            print 'X',X
#            print 'Y',Y
#            print 'Y_Err',Y_Err
            
            if(len(X) < len(Xs[i,:])):
                X = self.padFinalValue(X, len(Xs[i,:]))
                Y = self.padFinalValue(Y, len(Ys[i,:]))
                X_Err = self.padFinalValue(X_Err, len(X_Errs[i,:]))
                Y_Err = self.padFinalValue(Y_Err, len(Y_Errs[i,:]))
            Xs[i,:]     = X
            Ys[i,:]     = Y
            X_Errs[i,:]   = X_Err
            Y_Errs[i,:]   = Y_Err
                    
        xLims = [0,max(Xs[:,-1])*1.02]
        yLims = [min_-.1*abs(max_-min_),max_+.1*abs(max_-min_)]
        
        if not isOnCluster() and self.showSplash:
            if self.legend:
                #pl.legend(loc='lower right',b_to_anchor=(0, 0),fancybox=True,shadow=True, ncol=1, mode='')
                self.legend = pl.legend(fancybox=True,shadow=True, ncol=1, frameon=True,loc=(1.03,0.2))
                
                #pl.axes([0.125,0.2,0.95-0.125,0.95-0.2])
            pl.xlim(xLims)
            if min_ != max_: pl.ylim(yLims)
            X_axis_label = r'$\|A\theta - b\|$' if X_axis == 'Error' else X_axis
            Y_axis_label = r'$\|A\theta - b\|$' if Y_axis == 'Error' else Y_axis
                    
            pl.xlabel(X_axis_label,fontsize=16)
            pl.ylabel(Y_axis_label,fontsize=16)
        self.save(Y_axis,X_axis,Xs,Ys,X_Errs, Y_Errs)
        #if not isOnCluster and self.legend:
        #        # This is a hack so we can see it correctly during the runtime
        #        pl.legend(loc='lower right',fancybox=True,shadow=True, ncol=1, mode='')
    
    ## Takes an array and extends it to desired length, copying final value.
    def padFinalValue(self, arr, newLength):
        if len(arr) < newLength: # Pad remaining values with final value
            paddedArr = zeros(newLength)
            paddedArr[:len(arr)] = arr[:] # Beginning of array = original
            paddedArr[len(arr):] = arr[-1]; # Remainder = last value
            return paddedArr
        else: return arr
    
    def save(self,Y_axis,X_axis,Xs,Ys,X_Errs,Y_Errs):
        fullfilename = self.output_path + '/' +Y_axis+'-by-'+X_axis
        checkNCreateDirectory(fullfilename)
        # Store the numbers in a txt file
        f = open(fullfilename+'.txt','w')
        for i in range(self.exp_num):
            # Print in the standard error:
            print "======================"
            print "Algorithm: ", self.exp_paths[i]
            print "======================"
            print X_axis +': ' + pretty(Xs[i,:],'%0.0f')
            print Y_axis +': ' + pretty(Ys[i,:])
            print 'X Standard-Error:' + pretty(X_Errs[i,:])
            print 'Y Standard-Error: ' + pretty(Y_Errs[i,:])
            if self.prettyText:
                f.write("======================\n")
                f.write("Algorithm: " + self.labels[i] +"\n")
                f.write("======================\n")
                f.write(X_axis +': ' + pretty(Xs[i,:])+'\n')
                f.write(Y_axis +': ' + pretty(Ys[i,:])+'\n')
                f.write('X Standard-Error: ' + pretty(X_Errs[i,:])+'\n')
                f.write('Y Standard-Error: ' + pretty(Y_Errs[i,:])+'\n')
            else:
                savetxt(f,Xs[i,:], fmt='%0.4f', delimiter='\t')
                savetxt(f,Ys[i,:], fmt='%0.4f', delimiter='\t')
                savetxt(f,X_Errs[i,:], fmt='%0.4f', delimiter='\t')
                savetxt(f,Y_Errs[i,:], fmt='%0.4f', delimiter='\t')
        f.close()
        # Save the figure as pdf
        if not isOnCluster() and self.showSplash:
            if self.legend:
                self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1,bbox_extra_artists=(self.legend,), bbox_inches='tight')
            else:
                self.fig.savefig(fullfilename+'.pdf', transparent=True, pad_inches=.1, bbox_inches='tight')
            print "==================\nSaved Outputs at\n1. %s\n2. %s" % (fullfilename+'.txt',fullfilename+'.pdf')
    def hasResults(self,path):
        return len(glob.glob(os.path.join(path, '*-results.txt'))) >= self.minSamples
    def extract_time_95(self,M):
        # Find the time the algorithm required to reach to its 95% performance from the starting point
        # M is the matrix of the results
        return_row  = self.AXES.index('Return')
        time_row    = self.AXES.index('Time(s)')
        rows,cols   = M.shape
        target_col  = 1
        last_return = M[return_row,-1] 
        #print last_return
        while target_col < cols  and abs(M[return_row,target_col]-last_return) > (abs(last_return)*.05):
            #print  M[return_row,target_col], last_return, abs(M[return_row,target_col]-last_return), abs(last_return)*.05
            #print target_col, cols
            target_col += 1
            
#        print  M[return_row,target_col], last_return, abs(M[return_row,target_col]-last_return), abs(last_return)*.05
        return M[time_row,target_col-1]
    def showTime95(self):
        print "======================================="
        print " Time to reach 95% of last performance "
        print "======================================="
        for i in range(self.exp_num):
            print "%s: %0.3f+%0.3f Based on %d Samples" % (self.exp_paths[i], self.time_95_mean[i] , self.time_95_std_errs[i],self.samples[i])
