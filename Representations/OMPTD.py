######################################################
# Developed by Alborz Geramiard Jan 11th 2013 at MIT #
######################################################
# OMP-TD implementation based on ICML 2012 paper of Wakefield and Parr
# This implementation assumes an initial representation exists and the bag of features is the conjunctions of existing features
# As a results OMP-TD uses iFDD to represents its features, yet its discovery method is different compared to iFDD as it does not have to look at the fringe of the tree
# rather it looks through a predefined set of features
# iFDD initially will expand all the features in the bag
# The set of features used by OMPTD aside from the initial_features are represented by self.expandedFeatures
from Representation import *
from iFDD import *

class OMPTD(Representation):
    maxBatchDicovery = 0     # Maximum number of features to be expanded on each iteration
    batchThreshold  = 0      # Minimum threshold to add features
    selectedFeatures = []    # List of selected features. In this implementation initial features are selected initially by default
    remainingFeatures = None # Array of remaining features
    def __init__(self,domain,logger, initial_representation, discretization = 20,maxBatchDicovery = 1, batchThreshold = 0, bagSize = 100000, sparsify = False):
        self.iFDD_ONLINETHRESHOLD   = 1 # This is dummy since omptd will not use ifdd in the online fashion
        self.maxBatchDicovery       = maxBatchDicovery
        self.batchThreshold         = batchThreshold
        self.initial_representation = initial_representation                   
        self.iFDD                   = iFDD(domain,logger,self.iFDD_ONLINETHRESHOLD, initial_representation, sparsify = 0, discretization = discretization, useCache = 1)
        self.bagSize                = bagSize
        self.features_num           = self.initial_representation.features_num

        super(OMPTD,self).__init__(domain,logger,discretization)
        
        self.fillBag()
        self.totalFeatureSize       = self.bagSize
        self.selectedFeatures       = range(self.initial_representation.features_num) #Add initial features to the selected list
        self.remainingFeatures      = arange(self.features_num,self.bagSize) # Array of indicies of features that have not been selected
        self.show()            
        if self.logger:
            self.logger.log("Initial Representation:\t%s"% className(self.initial_representation))
            self.logger.log("Bag Size:\t\t%d"% self.bagSize)
            self.logger.log("Batch Threshold:\t\t%0.3f"% self.batchThreshold)
            self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
    def phi_nonTerminal(self,s):
        F_s                     = self.iFDD.phi_nonTerminal(s)
        return F_s[self.selectedFeatures]
    def show(self):
        if self.logger:
            self.logger.log('Features:\t\t%d' % self.features_num)
            self.logger.log('Remaining Bag Size:\t%d' % len(self.remainingFeatures))
    def calculateFullPhiNormalized(self,states):
        # In general for OMPTD it is faster to cashe the normalized phi matrix for all states for all features in one shot.
        # If states are changed this function should be called once to recalculate the phi matrix
        # TO BE FILLED
        p = len(states)
        self.fullphi = empty((p,self.totalFeatureSize))
        for i,s in enumerate(states):
            if not self.domain.isTerminal(s):
                self.fullphi[i,:] = self.iFDD.phi_nonTerminal(s)

        #Normalize features
        for f in arange(self.totalFeatureSize):
            phi_f               = self.fullphi[:,f]
            norm_phi_f          = linalg.norm(phi_f)    # L2-Norm of phi_f
            if norm_phi_f == 0: norm_phi_f = 1          # This helps to avoid divide by zero 
            self.fullphi[:,f]   =  phi_f/norm_phi_f
    def batchDiscover(self,td_errors, phi, states):
        # Discovers features using OMPTD
        # 1. Find the index of remaining features in the bag
        # 2. Calculate the inner product of each feature with the TD_Error vector
        # 3. Add the top maxBatchDicovery features to the selected features
        ###############
        # INPUT:
        # td_errors     p-by-1
        # phi           p-by-n
        # states        p-by-state-dim
        ###############
        # OUTOUT: Boolean indicating expansion of features
        ###############
        
        if len(self.remainingFeatures) == 0:
            # No More features to Expand
            return False

        SHOW_RELEVANCES = 0      # Plot the relevances 
        max_excitement  = -inf
        n               = self.features_num  #number of features
        p               = len(td_errors)     #Number of samples
        self.calculateFullPhiNormalized(states)
        
        relevances = zeros(len(self.remainingFeatures))
        for i,f in enumerate(self.remainingFeatures):
            phi_f           = self.fullphi[:,f]
            relevances[i]   = abs(dot(phi_f,td_errors))
        
        if SHOW_RELEVANCES:
            e_vec  = relevances.flatten()
            e_vec  = e_vec[e_vec != 0]
            e_vec  = sort(e_vec)
            pl.plot(e_vec,linewidth=3)
            pl.ioff()
            pl.show()
            pl.ion()

        #Sort based on relevances
        sortedIndices  = argsort(relevances)[::-1] # We want high to low hence the reverse: [::-1]
        max_relevance   = relevances[sortedIndices[0]];
        
        #Add top <maxDiscovery> features
        self.logger.log("OMPTD Batch: Max Relevance = %0.3f" % max_relevance)
        added_feature           = False
        to_be_deleted = [] # Record the indices of items to be removed 
        for j in arange(min(self.maxBatchDicovery,len(relevances))):
            max_index   = sortedIndices[j]
            f           = self.remainingFeatures[max_index]
            relevance   = relevances[max_index]
            if relevance > self.batchThreshold:
                self.logger.log('New Feature %d: %s, Relevance = %0.3f' % (self.features_num, str(sort(list(self.iFDD.getFeature(f).f_set))),relevances[max_index]))
                to_be_deleted.append(f)
                self.selectedFeatures.append(f)
                self.features_num += 1
                added_feature = True
            else:
                #Because the list is sorted, there is no use to look at the others
                break
        self.remainingFeatures = delete(self.remainingFeatures,to_be_deleted)
        return added_feature # A signal to see if the representation has been expanded or not
    def fillBag(self):
        # This function generates lists of potential features to be put in the bag each indicated by a list of initial features. The resulting feature is the conjunction of the features in the list
        # The potential list is expanded by traversing the tree in the BFS fashion untill the bagSize is reached.
        level_1_features        = arange(self.initial_representation.features_num)
        # We store the dimension corresponding to each feature so we avoid adding pairs of features in the same dimension
        level_1_features_dim = {}
        for i in arange(self.initial_representation.features_num):
            level_1_features_dim[i] = array([self.initial_representation.getDimNumber(i)])
            #print i,level_1_features_dim[i]
        level_n_features        = array(level_1_features)
        level_n_features_dim    = deepcopy(level_1_features_dim)
        new_id                  = self.initial_representation.features_num
        if self.logger: self.logger.log("Added %d size 1 features to the feature bag." % (self.initial_representation.features_num))

        # Loop over possible layers that conjunctions can be add. Notice that layer one was already built
        for f_size in arange(2,self.domain.state_space_dims+1):
            added = 0
            next_features = []
            next_features_dim = {}
            for f in level_1_features:
                f_dim = level_1_features_dim[f][0]
                for g in level_n_features:
                    g_dims = level_n_features_dim[g]
                    if not f_dim in g_dims:
                        added_new_feature = self.iFDD.inspectPair(f,g,inf) # We pass inf to make sure iFDD will add the combination of these two features
                        if added_new_feature:
                            #print '%d: [%s,%s]' % (new_id, str(f),str(g))
                            next_features.append(new_id)
                            next_features_dim[new_id] = g_dims + f_dim 
                            new_id += 1
                            added += 1
                            if new_id == self.bagSize:
                                return
            level_n_features        = next_features
            level_n_features_dim    = next_features_dim
            if self.logger: self.logger.log("Added %d size %d features to the feature bag." % (added, f_size))
        self.bagSize = new_id
if __name__ == '__main__':
    STDOUT_FILE         = 'out.txt'
    JOB_ID              = 1
    OUT_PATH            = 'Results/Temp'
#    logger              = Logger('%s/%d-%s'%(OUT_PATH,JOB_ID,STDOUT_FILE))
    logger              = Logger()
    discovery_threshold = 1
    #domain      = MountainCar()
    domain      = SystemAdministrator('/../Domains/SystemAdministratorMaps/20MachTutorial.txt')
    initialRep  = IndependentDiscretizationCompactBinary(domain,logger)
    rep         = iFDD(domain,logger,discovery_threshold,initialRep,debug=0,useCache=1)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    rep.showCache()
    print 'discover 0,20'
    phi_s = zeros(rep.features_num)
    phi_s[0] = 1
    phi_s[20] = 1
    rep.discover(phi_s, discovery_threshold+1)
    rep.showFeatures()
    rep.showCache()
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    rep.showCache()
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print 'Initial [0,20] => ',
    print rep.findFinalActiveFeatures([0,20])
    print 'discover 0,1,20'
    rep.inspectPair(1,rep.features_num-1, discovery_threshold+1)
    rep.showFeatures()
    rep.showCache()
    print 'Initial [0,1,20] => ',
    print rep.findFinalActiveFeatures([0,1,20])
    rep.showCache()
    print 'Initial [0,1,2,3,4,5,6,7,8,20] => ',
    print rep.findFinalActiveFeatures([0,1,2,3,4,5,6,7,8,20])
    rep.showCache()
    
    
    
    