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
# recently added basis function has norm <= batchThreshold, which
# Parr et al. used as 10^-5.

import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))

#from scipy.interpolate import Rbf
from Tools import *
from Domains import PitMaze
from Representation import *


class BEBF(Representation):
    debug                   = 0
    maxBatchDicovery        = 1     # Number of features to be expanded in the batch setting; here 1 since each BEBF will be identical on a given iteration
    svm_epsilon             = None  # from sklearn: "epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value."
    features                = []    # Array of pointers to feature functions, indexed by order created
    batchThreshold          = None  # Minimum value of feature relevance for the batch setting (10^-5 per Parr et al.
    initial_features_num    = 0     # Initial number of features, initialized in __init__
    def __init__(self,domain,logger, discretization = 20, debug = 0, batchThreshold = 10 ** -3, svm_epsilon = .1):
        self.setBinsPerDimension(domain,discretization)
        self.initial_features_num      = int(sum(self.bins_per_dim)) # Effectively initialize with IndependentDiscretization
        self.features_num           = self.initial_features_num # Starting number of features equals the above, changes during execution
       # self.features_num           = 0
        self.debug                  = debug
        self.svm_epsilon            = svm_epsilon
        self.batchThreshold          = batchThreshold
        self.addInitialFeatures()
        super(BEBF,self).__init__(domain,logger,discretization)
        self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
        self.logger.log("Norm Threshold:\t\t%0.3f"% self.batchThreshold)
    
        ## @return: a function object corresponding to the 
    def getFunctionApproximation(self,X,y):
        #bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0, epsilon = 0.0005) # support vector regression
                                                 # C = penalty parameter of error term, default 1
        bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0, epsilon = self.svm_epsilon)
        bebfApprox.fit(X,y)
        return bebfApprox
    
#    def getFunctionApproximation(self,X,y):
        #return Rbf(X,y,function='multiquadric') # function = gaussian
        #bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0) # support vector regression
                                                # C = penalty parameter of error term, default 1
#        gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
#        gp.fit(X, y)
#        return gp
#        bebfApprox.fit(X,y)
#        return bebfApprox
    
    
    def addInitialFeatures(self):
#        numDims = 1
#        numSamples = 100
#        y = random.randn(numSamples)
#        X = random.randn(numSamples,numDims)
#        self.features = array([self.getFunctionApproximation(X, y)])
        pass
    def phi_nonTerminal(self,s):
        F_s = zeros(self.features_num)
        F_s[self.activeInitialFeatures(s)]  = 1 # From IndependentDiscretization
        bebf_features_num = self.features_num - self.initial_features_num
        for features_ind, F_s_ind in enumerate(arange(bebf_features_num) + self.initial_features_num):
            F_s[F_s_ind] = self.features[features_ind].predict(s)
#        print 's,F_s',s,F_s
#        shout('F_s:',F_s)
        return F_s
    
    ## Adds new features based on the Bellman Error in batch setting.
    # @param td_errors: p-by-1 (How much error observed for each sample)
    # @param phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
    # @param s: List of states corresponding to each td_error in td_errors (note that the same state may appear multiple times because of different actions taken while there)
    # self.batchThreshold is threshold below which no more BEBFs are added.
    def batchDiscover(self,td_errors, all_phi_s, s):
        # need states here instead?
        addedFeature = False
        # PLACEHOLDER for norm of function
        norm = max(abs(td_errors))# Norm of function
        for j in arange(self.maxBatchDicovery):
#            print 's'
#            print 'tderr',td_errors[0:20]
            self.features.append(self.getFunctionApproximation(s, td_errors))
            if self.debug:
                shout('s',s)
                shout('td_errors',td_errors)
            self.logger.log('Norm: %0.4f' % norm)
            if norm > self.batchThreshold:
#                print 'added feature'
                self.addNewWeight()
                addedFeature        = True
                self.features_num   += 1
                self.logger.log('Added feature. \t %d total feats' % self.features_num)
            else: 
                break
        return addedFeature
    
if __name__ == '__main__':
    STDOUT_FILE         = 'out.txt'
    JOB_ID              = 1
    OUT_PATH            = 'Results/Temp'
#    logger              = Logger('%s/%d-%s'%(OUT_PATH,JOB_ID,STDOUT_FILE))
    logger              = Logger()
    discovery_threshold = 1
    domain      = PitMaze()
    rep         = BEBF(domain,logger,debug=1,batchThreshold = 10 ** -5)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    print 'initial features'
    print rep.features_num,'---',rep.features
    s           = domain.s0()
    a           = domain.possibleActions(s)
    a = a[0]
    r,ns,terminal   = domain.step(s, a)
    print 'step 2 r,ns',r,ns
    
    
    