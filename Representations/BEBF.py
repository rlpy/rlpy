######################################################
#       Robert H. Klein, Dec. 19, 2012 at MIT        #
######################################################
#
# REQUIRES the implementation of locally-weighted
# projection regression (LWPR), available at:
# http://wcms.inf.ed.ac.uk/ipab/slmc/research/software-lwpr
#
# Parameters set according to:
# Parr et al. 'Analyzing Feature Generation for Function Approximation'
#
# Bellman-Error Basis Function Representation.
# 1. Initial basis function based on immediate reward.
# 2. Evaluate r + Q(s', \pi{s'}) - Q(s,a) for all samples.
# 3. Train function approximator on bellman error of present solution above
# 4. Add the above as a new basis function.
# 5. Repeat the process using the new basis until the most
# recently added basis function has norm <= normThreshold, which
# Parr et al. used as 10^-5.

#from scipy.interpolate import Rbf
from sklearn import svm


class BEBF(Representation):
    debug                   = 0
#    memoized_phi            = {}    # dictionary mapping each feature index 
    featureIndex2feature    = {}    # dictionary mapping each feature index (ID) to its feature (function) object
    useCache                = 0     # this should only increase speed. If results are different something is wrong
    maxBatchDicovery        = 0     # Number of features to be expanded in the batch setting
    normThreshold           = 0     # Minimum value of feature relevance for the batch setting (10^-5 per Parr et al.)
    def __init__(self,domain,logger, discretization = 20, debug = 0,maxBatchDicovery = 1, normThreshold = 10 ** -5):
        self.setBinsPerDimension(domain,discretization)
        self.features_num           = int(sum(self.bins_per_dim))
        self.debug                  = debug
        self.useCache               = useCache
        self.maxBatchDicovery       = maxBatchDicovery
        self.normThreshold          = normThreshold
        self.addInitialFeatures()
        super(BEBF,self).__init__(domain,logger,discretization)
        self.logger.log("Threshold:\t\t%0.3f" % self.discovery_threshold)
        self.logger.log("Cached:\t\t\t%d"% self.useCache)
        self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
        self.logger.log("Norm Threshold:\t\t%0.3f"% self.normThreshold)
    ## Adds an initial basis function estimated using only immediate reward.
    
        ## @return: a function object corresponding to the 
    def getFunctionApproximation(self,X,y):
        #return Rbf(X,y,function='multiquadric') # function = gaussian
        bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0) # support vector regression
                                                 # C = penalty parameter of error term, default 1
        return bebfApprox.fit(X,y)
    
    def addInitialFeatures(self):
        for i in range(self.features_num):
            feature = iFDD_feature(i)
            #shout(self,self.iFDD_features[ImmutableSet([i])].index)
            self.iFDD_features[ImmutableSet([i])] = feature
            self.featureIndex2feature[feature.index] = feature
    def phi_nonTerminal(self,s):
        # Based on Tuna's Master Thesis 2012
        F_s                     = zeros(self.features_num,'bool')
        activeIndecies          = Set(self.activeInitialFeatures(s))
        if self.useCache:
            finalActiveIndecies     = self.cache.get(ImmutableSet(activeIndecies))
            if finalActiveIndecies is None:        
                # run regular and update the cache
                finalActiveIndecies     = self.findFinalActiveFeatures(activeIndecies)
        else:
            finalActiveIndecies         = self.findFinalActiveFeatures(activeIndecies)
        F_s[finalActiveIndecies] = 1
        return F_s
    
    self.setBinsPerDimension(domain,discretization)
    
    ## Adds new features based on the Bellman Error in batch setting.
    # @param td_errors: p-by-1 (How much error observed for each sample)
    # @param phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
    # self.normThreshold is threshold below which no more BEBFs are added.
    def batchDiscover(self,td_errors, all_phi_s):
        # need states here instead?
        for j in range(self.maxBatchDicovery):
            newFeature = self.getFunctionApproximation(phi, td_errors)
            norm = # Norm of function
            if norm > self.batchThreshold:
                print 'Added Feature'
                added_feature = True
        