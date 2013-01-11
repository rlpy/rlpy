# TODO: Merge update weight with the parent's function (AddNewWeight or sth)
######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
import sys, os
from Queue import PriorityQueue
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domains import *
from Representation import *
from IndependentDiscretization import *
from IndependentDiscretizationCompactBinary import *

class iFDD_feature(object):
    index   = None        #Unique index Corresponding to its location in a vector (0 based!)
    f_set   = None        #A set of basic features that this feature corresponds to
    p1      = None        #Parent 1 feature index *Basic features have -1 for both nodes
    p2      = None        #Parent 2 feature index *Basic features have -1 for both nodes
    def __init__(self,potential):
        if isinstance(potential,iFDD_potential):
            self.index  = potential.index
            self.f_set  = deepcopy(potential.f_set)
            self.p1     = potential.p1
            self.p2     = potential.p2
        else:
            # This is the case where a single integer is passed to generate an initial feature
            index           = potential
            self.index      = index
            self.f_set      = frozenset([index])
            self.p1         = -1
            self.p2         = -1
    def show(self):
        printClass(self)
class iFDD_potential(object):
    relevance = None     # Sum of errors corresponding to this feature
    f_set     = None     # Set of features it corresponds to [Immutable]
    index     = None     # If this feature has been discovered set this to its index else 0
    p1        = None     # Parent 1 feature index
    p2        = None     # Parent 2 feature index
    count     = None     # Number of times this feature was visited
    def __init__(self,f_set,relevance,parent1,parent2):
        self.f_set      = deepcopy(f_set)
        self.index      = -1 # -1 means it has not been discovered yet
        self.p1         = parent1
        self.p2         = parent2
        self.relevance  = relevance
        self.count      = 1
    def show(self):
        printClass(self)
class iFDD(Representation):
    discovery_threshold     = None  # psi in the paper
    sparsify                = None  # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    iFDD_features           = {}    # dictionary mapping initial feature sets to iFDD_feature 
    iFDD_potentials         = {}    # dictionary mapping initial feature sets to iFDD_potential
    featureIndex2feature    = {}    # dictionary mapping each feature index (ID) to its feature object
    debug                   = 0     # Print more stuff
    cache                   = {}    # dictionary mapping  initial active feature set phi_0(s) to its corresponding active features at phi(s). Based on Tuna's Trick to speed up iFDD
    useCache                = 0     # this should only increase speed. If results are different something is wrong
    maxBatchDicovery        = 0     # Number of features to be expanded in the batch setting
    batchThreshold          = 0     # Minimum value of feature relevance for the batch setting 
    iFDDplus                = 1     # ICML 11 iFDD would add sum of abs(TD-errors) while the iFDD plus uses the abs(sum(TD-Error))/sqrt(potential feature presence count)    
    sortediFDDFeatures      = None  # This is a priority queue based on the size of the features (Largest -> Smallest). For same size features, tt is also sorted based on the newest -> oldest. Each element is the pointer to feature object.
    initial_Representation  = None  # A Representation that provides the initial set of features for iFDD 
    maxRelevance            = -inf  # Helper parameter to get a sense of appropriate threshold on the relevance for discovery
    def __init__(self,domain,logger,discovery_threshold, initial_Representation, sparsify = True, discretization = 20,debug = 0,useCache = 0,maxBatchDicovery = 1, batchThreshold = 0):
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.setBinsPerDimension(domain,discretization)
        self.features_num           = initial_Representation.features_num
        self.debug                  = debug
        self.useCache               = useCache
        self.maxBatchDicovery       = maxBatchDicovery
        self.batchThreshold         = batchThreshold
        self.sortediFDDFeatures     = PriorityQueueWithNovelty()
        self.initial_Representation = initial_Representation
        self.addInitialFeatures()
        super(iFDD,self).__init__(domain,logger,discretization)
        if self.logger:
            self.logger.log("Initial Representation:\t%s"% className(self.initial_Representation))
            self.logger.log("Plus:\t\t\t%d" % self.iFDDplus)
            self.logger.log("Sparsify:\t\t%d"% self.sparsify)
            self.logger.log("Cached:\t\t\t%d"% self.useCache)
            self.logger.log("Online Threshold:\t%0.3f" % self.discovery_threshold)
            self.logger.log("Batch Threshold:\t\t%0.3f"% self.batchThreshold)
            self.logger.log("Max Batch Discovery:\t%d"% self.maxBatchDicovery)
    def phi_nonTerminal(self,s):
        # Based on Tuna's Master Thesis 2012
        F_s                     = zeros(self.features_num,'bool')
        F_s_0                   = self.initial_Representation.phi_nonTerminal(s)
        activeIndices           = where(F_s_0 != 0)[0]
        if self.useCache:
            finalActiveIndices     = self.cache.get(frozenset(activeIndices))
            if finalActiveIndices is None:        
                # run regular and update the cache
                finalActiveIndices     = self.findFinalActiveFeatures(activeIndices)
        else:
            finalActiveIndices         = self.findFinalActiveFeatures(activeIndices)
        F_s[finalActiveIndices] = 1
        return F_s
    def findFinalActiveFeatures(self,intialActiveFeatures):
        # Given the active indices of phi_0(s) find the final active indices of phi(s) based on discovered features
        finalActiveFeatures = []
        k = len(intialActiveFeatures)
        initialSet          = set(intialActiveFeatures)
        if 2**k <= self.features_num:
            # k can be big which can cause this part to be very slow
            # if k is large then find active features by enumerating on the 
            # discovered features.  
            for candidate in powerset(initialSet,ascending=0):
                if len(initialSet) == 0:
                    break # No more initial features to be mapped to extended ones
                
                if initialSet.issuperset(set(candidate)): # This was missing from ICML 2011 paper algorithm. Example: [0,1,20], [0,20] is discovered, but if [0] is checked before [1] it will be added even though it is already covered by [0,20]
                    feature = self.iFDD_features.get(frozenset(candidate))
                    if feature != None:
                        finalActiveFeatures.append(feature.index)
                        if self.sparsify:
                            #print "Sets:", initialSet, feature.f_set 
                            initialSet   = initialSet - feature.f_set
                            #print "Remaining Set:", initialSet
        else:
            #print "********** Using Alternative: %d > %d" % (2**k, self.features_num)
            # Loop on all features sorted on their size and then novelty and activate features
            for feature in self.sortediFDDFeatures.toList():
                if len(initialSet) == 0:
                    break # No more initial features to be mapped to extended ones

                if initialSet.issuperset(set(feature.f_set)): 
                    finalActiveFeatures.append(feature.index)
                    if self.sparsify:
                        #print "Sets:", initialSet, feature.f_set 
                        initialSet   = initialSet - feature.f_set
                        #print "Remaining Set:", initialSet
            
        if self.useCache:
            self.cache[frozenset(intialActiveFeatures)] = finalActiveFeatures
        return finalActiveFeatures     
    def discover(self,phi_s,td_error):            
        activeFeatures = phi_s.nonzero()[0] # Indices of non-zero elements of vector phi_s
        for g_index,h_index in combinations(activeFeatures,2):
            self.inspectPair(g_index,h_index,td_error)
    def inspectPair(self,g_index,h_index,td_error):
        # Inspect feature f = g union h where g_index and h_index are the indices of features g and h        
        # If the relevance is > Threshold add it to the list of features
        g  = self.featureIndex2feature[g_index].f_set 
        h  = self.featureIndex2feature[h_index].f_set
        f           = g.union(h)
        feature     = self.iFDD_features.get(f)
        if feature is None:
            #Look it up in potentials
            potential = self.iFDD_potentials.get(f)
            if potential is None:
                # Generate a new potential and put it in the dictionary
                potential = iFDD_potential(f,td_error,g_index,h_index)
                self.iFDD_potentials[f] = potential
            else:
                if self.iFDDplus:
                    potential.relevance += td_error
                else:
                    potential.relevance += abs(td_error)
                potential.count     += 1
            # Check for discovery
            relevance = potential.relevance if not self.iFDDplus else abs(potential.relevance/sqrt(potential.count)) 
            if relevance > self.discovery_threshold:
                self.addFeature(potential)
                self.maxRelevance = -inf
            else:
                self.updateMaxRelevance(relevance)

    def show(self):
        self.showFeatures()
        self.showPotentials()
        self.showCache()
    def updateWeight(self,p1_index,p2_index):
        # Add a new weight corresponding to the new added feature for all actions.
        # The new weight is set to zero if sparsify = False, and equal to the sum of weights corresponding to the parents if sparsify = True
        a = self.domain.actions_num
        f = self.features_num-1 # Number of feature before adding the new one
        if self.sparsify:
            newElem = (self.theta[p1_index::f] + self.theta[p2_index::f]).reshape((-1,1)) 
        else:
            newElem =  None
        self.theta      = addNewElementForAllActions(self.theta,a,newElem)
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!
    def addInitialFeatures(self):
        for i in arange(self.initial_Representation.features_num):
            feature = iFDD_feature(i)
            #shout(self,self.iFDD_features[frozenset([i])].index)
            self.iFDD_features[frozenset([i])] = feature
            self.featureIndex2feature[feature.index] = feature
            priority = 1 # priority is 1/number of initial features corresponding to the feature
            self.sortediFDDFeatures.push(priority,feature)
    def addFeature(self,potential):
        # Add it to the list of features
        potential.index                     = self.features_num # Features_num is always one more than the max index (0-based)
        self.features_num                   += 1
        feature                             = iFDD_feature(potential)
        self.iFDD_features[potential.f_set] = feature
        #Expand the size of the theta
        self.updateWeight(feature.p1,feature.p2)
        # Update the index to feature dictionary
        self.featureIndex2feature[feature.index] = feature
        # Update the sorted list of features
        priority = 1/(len(potential.f_set)*1.) # priority is 1/number of initial features corresponding to the feature
        self.sortediFDDFeatures.push(priority,feature)

        #If you use cache, you should invalidate entries that their initial set contains the set corresponding to the new feature 
        if self.useCache:
            for initialActiveFeatures in self.cache.keys():
                if initialActiveFeatures.issuperset(feature.f_set):
                    self.cache.pop(initialActiveFeatures)
        
        if self.debug: self.show()
    def batchDiscover(self,td_errors, phi, states):
        # Discovers features using iFDD in batch setting.
        # TD_Error: p-by-1 (How much error observed for each sample)
        # phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
        # self.batchThreshold is the minimum relevance value for the feature to be expanded
        SHOW_HISTOGRAM  = 0      #Shows the histogram of relevances 
        max_excitement  = 0
        maxDiscovery    = self.maxBatchDicovery
        n               = self.features_num #number of features
        p               = len(td_errors)     #Number of samples
        counts          = zeros((n,n))
        relevances      = zeros((n,n))
        added_feature   = True
        for i in arange(p):
            phiphiT     = outer(phi[i,:],phi[i,:])
            if self.iFDDplus:
                relevances  += phiphiT*td_errors[i]
            else:
                relevances  += phiphiT*abs(td_errors[i])
            counts      += phiphiT
        
        #Remove Diagonal and upper part of the relevances as they are useless
        relevances      = triu(relevances,1)
        non_zero_index  = nonzero(relevances)
        if self.iFDDplus:
            relevances[non_zero_index] = divide(abs(relevances[non_zero_index]), sqrt(counts[non_zero_index])) #Calculate relevances based on theoretical results of ICML 2013 potential submission
        else:
            relevances[non_zero_index] = relevances[non_zero_index] #Based on Geramifard11_ICML Paper

        if SHOW_HISTOGRAM:
            e_vec  = relevances.flatten()
            e_vec  = e_vec[e_vec != 0]
            e_vec  = sort(e_vec)
            drawHist(e_vec)
            pl.show()

        #Find indexes to non-zero excited pairs
        (F1, F2)            = relevances.nonzero() # F1 and F2 are the parents of the potentials
        relevances          = relevances[F1,F2]
        if len(relevances) == 0:
            # No feature to add
            self.logger.log("iFDD Batch: Max Relevance = 0") 
            return False
        #Sort based on relevances
        sortedIndices  = argsort(relevances)[::-1] # We want high to low hence the reverse: [::-1]
        max_relevance   = relevances[sortedIndices[0]];
        #Add top <maxDiscovery> features
        self.logger.log("iFDD Batch: Max Relevance = %0.3f" % max_relevance)
        added_feature = False
        for j in arange(min(maxDiscovery,len(relevances))):
            max_index   = sortedIndices[j]
            f1          = F1[max_index]
            f2          = F2[max_index]
            relevance   = relevances[max_index]
            if relevance > self.batchThreshold:
                g  = self.featureIndex2feature[f1].f_set 
                h  = self.featureIndex2feature[f2].f_set
                self.logger.log('New Feature %d: %s, Relevance = %0.3f' % (j+1, str(sorted([s for s in g.union(h)])),relevances[max_index]))
                self.inspectPair(f1, f2, inf)
                added_feature = True
            else:
                #Because the list is sorted, there is no use to look at the others
                break
        return added_feature # A signal to see if the representation has been expanded or not
    def showFeatures(self):
        print "Features:"
        print join(["-"]*30)
        print " index\t| f_set\t| p1\t| p2\t | Weights (per action)"
        print join(["-"]*30)
        for feature in self.sortediFDDFeatures.toList():
        #for feature in self.iFDD_features.itervalues():
            #print " %d\t| %s\t| %s\t| %s\t| %s" % (feature.index,str(list(feature.f_set)),feature.p1,feature.p2,str(self.theta[feature.index::self.features_num]))
            print " %d\t| %s\t| %s\t| %s\t| Omitted" % (feature.index,str(list(feature.f_set)),feature.p1,feature.p2)
    def showPotentials(self):
        print "Potentials:"
        print join(["-"]*30)
        print " index\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.index,str(list(potential.f_set)),potential.relevance,potential.count,potential.p1,potential.p2)
    def showCache(self):
        if self.useCache: 
            print "Cache:"
            print join(["-"]*30)
            print " initial\t| Final"
            print join(["-"]*30)
            for initial,active in self.cache.iteritems():
                print " %s\t| %s" % (str(list(initial)), active)
    def updateMaxRelevance(self, newRelevance):
        # Update a global max relevance and outputs it if it is updated
        if self.maxRelevance < newRelevance:
            self.maxRelevance = newRelevance
            self.logger.log('iFDD: New Max Relevance: %0.3f' % newRelevance)
            
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
    
    
    
    