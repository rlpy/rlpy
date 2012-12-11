######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
# Assuming Linear Function approximator Family
from Tools import *
class Representation(object):
    DEBUG           = 0
    theta           = None  #Linear Weights
    domain          = None  #Link to the domain object 
    features_num    = None  #Number of features
    discretization  = 0     #Number of bins used for discretization for each continuous dimension  
    bins_per_dim = None     #Number of possible states per dimension [1-by-dim]
    agg_states_num  = None  #Number of aggregated states based on the discretization. If the represenation is adaptive set it to the best resolution possible  
    logger = None           # Object for capturing output text in a file
    def __init__(self,domain,logger,discretization = 20):
        # See if the child has set important attributes  
        for v in ['features_num']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.setBinsPerDimension(domain,discretization)
        self.domain = domain
        self.discretization = discretization
        self.theta  = zeros(self.features_num*self.domain.actions_num) 
        self.agg_states_num = prod(self.bins_per_dim.astype('uint64'))
        self.logger = logger
        self.logger.line()
        self.logger.log("Representation:\t\t%s" % className(self))
        if len(self.domain.continuous_dims): self.logger.log("Discretization:\t\t%d"% self.discretization)
        self.logger.log("Starting Features:\t%d"% self.features_num)
        self.logger.log("Aggregated States:\t%d"% self.agg_states_num)
    def V(self,s, phi_s = None):
        #Returns the value of a state
        if phi_s is None: phi_s = self.phi(s)
        AllQs,A   = self.Qs(s,phi_s)
        V       = max(AllQs)
        return V
    def Qs(self,s, phi_s = None):
    #Returns two arrays
    # Q: array of Q(s,a)
    # A: Corresponding array of action numbers
    # If phi_s is given, it uses that to speed up the process
        A = self.domain.possibleActions(s)
        if phi_s is None: phi_s   = self.phi(s)
        return array([self.Q(s,a,phi_s) for a in A]), A     
    def Q(self,s,a,phi_s = None):
        #Returns the state-action value
        if len(self.theta) > 0:
            return dot(self.phi_sa(s,a, phi_s),self.theta)
        else:
            return 0.0
    def phi(self,s):
        #Returns the phi(s)
        if self.domain.isTerminal(s):
            return zeros(self.features_num,'bool')
        else:
            return self.phi_nonTerminal(s)
    def phi_sa(self,s,a, phi_s = None):
        #Returns the feature vector corresponding to s,a (we use copy paste technique (Lagoudakis & Parr 2003)
        #If phi_s is passed it is used to avoid phi_s calculation
        if phi_s is None: phi_s = self.phi(s)

        phi_sa      = zeros(self.features_num*self.domain.actions_num)
        ind_a       = range(a*self.features_num,(a+1)*self.features_num)
        phi_sa[ind_a] = phi_s
        return phi_sa
        # Use of Kron is slower!
        #A = zeros(self.domain.actions_num)
        #A[a] = 1
        #F_sa = kron(A,F_s)
    def addNewWeight(self):
        # Add a new 0 weight corresponding to the new added feature for all actions.
        self.theta      = addNewElementForAllActions(self.theta,self.domain.actions_num)
    def hashState(self,s,):
        #returns a unique idea by calculating the enumerated number corresponding to a state
        # it first translate the state into a binState (bin number corresponding to each dimension)
        # it then map the binstate to a an integer
        ds = self.binState(s)
        return vec2id(ds,self.bins_per_dim)
    def setBinsPerDimension(self,domain,discretization):
        # Set the number of bins for each dimension of the domain (continuous spaces will be slices using the discritization parameter)
        self.bins_per_dim = zeros(domain.state_space_dims,uint16)
        for d in arange(domain.state_space_dims):
             if d in domain.continuous_dims:
                 self.bins_per_dim[d] = discretization
             else:
                 self.bins_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]
    def binState(self,s):
        # Given a state it returns a vector with the same dimensionality of s
        # each element of the returned valued is the zero-indexed bin number corresponding to s
        # note that s can be continuous.  
        # 1D examples: 
        # s = 0, limits = [-1,5], bins = 6 => 1
        # s = .001, limits = [-1,5], bins = 6 => 1
        # s = .4, limits = [-.5,.5], bins = 3 => 2
        if isinstance(s,int): return s 
        bs  = empty(len(s),'uint16')
        
        for d in arange(self.domain.state_space_dims):
            bs[d] = binNumber(s[d],self.bins_per_dim[d],self.domain.statespace_limits[d,:])
        return bs
    def printAll(self):
        printClass(self)
    def bestActions(self,s, phi_s = None):
    # Given a state returns the best action possibles at that state
    # If phi_s is given it is used to speed up
        Qs, A = self.Qs(s,phi_s)
        # Find the index of best actions
        ind   = findElemArray1D(Qs,Qs.max())
        if self.DEBUG:
            print 'State:',s
            print '======================================='
            for i in arange(len(A)):
                print 'Action %d, Q = %0.5f' % (A[i], Qs[i])
            print '======================================='
            print 'Best:', A[ind], 'MAX:', Qs.max()
            #raw_input()
        return A[ind]

#    def discretized(self,s):
#        ds = s.copy()
#        for dim in self.domain.continuous_dims:
#                ds[dim] = closestDiscretization(ds[dim],self.discretization,self.domain.statespace_limits[dim][:]) 
#        return ds
    def bestAction(self,s, phi_s = None):
        # return an action among the best actions uniformly randomly:
        bestA = self.bestActions(s,phi_s)
        if len(bestA) > 1:
            return randSet(bestA)
        else:
            return bestA[0]
    def phi_nonTerminal(self,s):
            # This is the actual function that each representation should fill
            # if state is terminal the feature vector is always zero!
            abstract
    def activeInitialFeatures(self,s):
        #return the index of active initial features based on bins on each dimensions
        bs          = self.binState(s)
        shifts      = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index       = bs+shifts
        return      index.astype('uint32')
    