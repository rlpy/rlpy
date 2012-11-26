######################################################
# Developed by Alborz Geramiard Nov 14th 2012 at MIT #
######################################################
# iFDD implementation based on ICML 2011 paper
import sys, os
#Add all paths
sys.path.insert(0, os.path.abspath('..'))
from Tools import *
from Domains import MountainCar
from Representation import *

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
            self.f_set      = ImmutableSet([index])
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
    discovery_threshold     = None # psi in the paper
    sparsify                = None # boolean specifying the use of the trick mentioned in the paper so that features are getting sparser with more feature discoveries (i.e. use greedy algorithm for feature activation)
    iFDD_features           = {}   # dictionary mapping initial feature sets to iFDD_feature 
    iFDD_potentials         = {}   # dictionary mapping initial feature sets to iFDD_potential
    featureIndex2feature    = {}   # dictionary mapping each feature index (ID) to its feature object
    def __init__(self,domain,discovery_threshold, sparsify = True, discretization = 20):
        self.discovery_threshold    = discovery_threshold
        self.sparsify               = sparsify
        self.setBinsPerDimension(domain,discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.add_initialFeatures()
        super(iFDD,self).__init__(domain,discretization)
    def phi_nonTerminal(self,s): #refer to Algorithm 2 in [Geramifard et al. 2011 ICML]
        F_s                     = zeros(self.features_num,'bool')
        activeIndecies          = ImmutableSet(self.activeInitialFeatures(s))
        for candidate in powerset(activeIndecies,ascending=0):
            
            if len(activeIndecies) == 0:
                break
            
            feature = self.iFDD_features.get(ImmutableSet(candidate))
            if feature != None:
                F_s[feature.index]  = 1
                if self.sparsify:
                    activeIndecies   = activeIndecies - feature.f_set
        return F_s 
    def discover(self,phi_s,td_error):            
        activeFeatures = phi_s.nonzero()[0] # Indecies of non-zero elements of vector phi_s
        for pair in combinations(activeFeatures,2):
            g,h         = pair
            g_initials  = self.featureIndex2feature[g].f_set 
            h_initials  = self.featureIndex2feature[h].f_set
            f           = g_initials.union(h_initials)
            feature     = self.iFDD_features.get(f)
            if feature is None:
                #Look it up in potentials
                potential = self.iFDD_potentials.get(f)
                if potential is None:
                    # Generate a new potential and put it in the dictionary
                    potential = iFDD_potential(f,td_error,g,h)
                    self.iFDD_potentials[f] = potential
                else:
                    potential.relevance += td_error
                    potential.count     += 1
                # Check for discovery
                if abs(potential.relevance/sqrt(potential.count)) > self.discovery_threshold:
                    self.addFeature(potential)
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
    def show(self):
        print "Features:"
        print join(["-"]*30)
        print " index\t| f_set\t| p1\t| p2\t | Weights"
        print join(["-"]*30)
        for _,feature in self.iFDD_features.iteritems():
            print " %d\t| %s\t| %s\t| %s\t| %s" % (feature.index,str(list(feature.f_set)),feature.p1,feature.p2,str(self.theta[feature.index::self.features_num]))
        print "Potentials:"
        print join(["-"]*30)
        print " index\t| f_set\t| relevance\t| count\t| p1\t| p2"
        print join(["-"]*30)
        for _,potential in self.iFDD_potentials.iteritems():
            print " %d\t| %s\t| %0.2f\t| %d\t| %s\t| %s" % (potential.index,str(list(potential.f_set)),potential.relevance,potential.count,potential.p1,potential.p2)
    def saveRepStat(self):
        pass
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
    def add_initialFeatures(self):
        for i in range(self.features_num):
            feature = iFDD_feature(i)
            #shout(self,self.iFDD_features[ImmutableSet([i])].index)
            self.iFDD_features[ImmutableSet([i])] = feature
            self.featureIndex2feature[feature.index] = feature
    def batchDiscover(self,td_error, phi, maxDiscovery = 1):
        # Discovers features using iFDD in batch setting.
        # TD_Error: p-by-1 (How much error observed for each sample)
        # phi: n-by-p features corresponding to all samples (each column corresponds to one sample)
        
        SHOW_HISTOGRAM  = 1      #Shows the histogram of relevances 
        max_excitement  = 0
        n               = self.features_num #number of features
        p               = len(td_error)     #Number of samples
        counts          = zeros((n,n))
        relevances      = zeros((n,n))
        
        for i in range(p):
            phiphiT     = outer(phi[:,i],phi[:,i])
            relevances  += phiphiT*td_error(i)
            counts      += phiphiT
        
        #Remove Diagonal and upper part of the relevances as they are useless
        relevances      = triu(relevances,1)
        non_zero_index  = nonzero(relevances)
        relevances[non_zero_index] = divide(abs(relevances[non_zero_index]), sqrt(counts[non_zero_index])) #Calculate relevances based on theoretical results of ICML 2013 potential submission 

        if SHOW_HISTOGRAM:
            e_vec  = relevances.flatten()
            e_vec  = e_vec[e_vec != 0]
            e_vec  = sort(e_vec)
            drawHist(e_vec)
            pl.show(block=False)

        #Find indexes to non-zero excited pairs
        (F1, F2)            = relevances.nonzero() # F1 and F2 are the parents of the potentials
        relevances          = relevances[F1,F2]
        #Sort based on relevances
        sortedIndecies  = argsort(relevances).reverse() # We want high to low hence the reverse
        max_relevance   = relevances(sortedIndecies[0]);

        #Add top <maxDiscovery> features
        for j in range(min(maxDiscovery,len(relevances))):
            max_index   = sortedIndecies[j]
            f1          = F1[max_index]
            f2          = F2[max_index]
            relevance   = relevances[max_index];
            fprintf('Added Feature [%d,%d]\n',F1,F2);
            # find initial sets for f1 and f2
            f1_initial = self.featureID2feature[f1].f_set
            f2_initial = self.featureID2feature[f2].f_set
            
#            % Different actions may translate into the same features! You
#            % should add more code if you want excactly maxDiscovery number
#            % of features added on each iteration           
#        else
#            break;
#        end
#        
#    end
#    
#    ending_feature_size = length(agent.theta);
#    expanded_rep = (ending_feature_size ~= starting_feature_size);
#end   

if __name__ == '__main__':
    discovery_threshold = 1
    domain      = MountainCar()
    rep         = iFDD(domain,discovery_threshold)
    rep.theta   = arange(rep.features_num*domain.actions_num)*10
    rep.show()
    print 'discover 0,20'
    phi_s = zeros(rep.features_num)
    phi_s[0] = 1
    phi_s[20] = 1
    rep.discover(phi_s, discovery_threshold+1)
    rep.show()
    print rep.theta
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print 'discover 0,1,20'
    phi_s = zeros(rep.features_num)
    phi_s[1] = 1
    phi_s[40] = 1
    rep.discover(phi_s, discovery_threshold+1)
    rep.show()
    print rep.theta
    s = domain.statespace_limits[:,0]
    print"s=",s
    print"Active Initial=", rep.activeInitialFeatures(s)
    print"Phi(s)=", rep.phi_nonTerminal(s).astype(uint8)
    