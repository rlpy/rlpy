######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
# Assuming Linear Function approximator Family
from Tools import *
class Representation(object):
    DEBUG           = 0
    theta           = None  #Linear Weights
    domain          = None  #Link to the domain object 
    hashed_s        = None  #Always remember the s corresponding to the last state
    hashed_phi      = None  #Always remember the phi corresponding to the last state
    features_num    = None  #Number of features
    discretization  = 0     #Number of bins used for discretization for each continuous dimension  
    bins_per_dim = None  #Number of possible states per dimension [1-by-dim]
    agg_states_num  = None  #Number of aggregated states based on the discretization. If the represenation is adaptive set it to the best resolution possible  
    def __init__(self,domain,discretization = 20):
        # See if the child has set important attributes  
        for v in ['features_num']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.setBinsPerDimension(domain,discretization)
        self.domain = domain
        self.discretization = discretization
        self.theta  = zeros(self.features_num*self.domain.actions_num) 
        self.agg_states_num = prod(self.bins_per_dim.astype('uint64'))
        print join(["-"]*30)
        print "Representation:\t\t", className(self)
        print "Discretization:\t\t", self.discretization
        print "Starting Features:\t", self.features_num
        print "Aggregated States:\t", self.agg_states_num
    def V(self,s):
        #Returns the value of a state
        AllQs   = self.Qs(s)
        V       = max(allQs)
    def Qs(self,s):
    #Returns two arrays
    # Q: array of Q(s,a)
    # A: Corresponding array of action numbers
        A = self.domain.possibleActions(s)
        return array([self.Q(s, a) for a in A]), A     
    def Q(self,s,a):
        #Returns the state-action value
        if len(self.theta) > 0: 
            return dot(self.phi_sa(s,a),self.theta)
        else:
            return 0.0
    def fastPhi(self,s):
        #Returns the feature corresponding to the state s using the hash
        # Check bounds
        if self.hashed_s is not None and array_equal(s,self.hashed_s):
            return self.hashed_phi
        else:
            self.hashed_s = s
            self.hashed_phi = self.phi(s) 
            return self.hashed_phi
    def phi(self,s):
        #Returns the phi(s)
        abstract
    def phi_sa(self,s,a):
        #Returns the feature vector corresponding to s,a (we use copy paste technique (Lagoudakis & Parr 2003)
        F_s         = self.fastPhi(s)
        F_sa        = zeros(self.features_num*self.domain.actions_num)  
        ind_a       = range(a*self.features_num,(a+1)*self.features_num)
        F_sa[ind_a] = F_s
        return F_sa
    def addNewWeight(self):
        # Add a new 0 weight corresponding to the new added feature for all actions.
        x               = self.theta.reshape(self.domain.actions_num,-1) # -1 means figure the other dimension yourself
        x               = hstack((x,zeros((self.domain.actions_num,1))))
        self.theta      = x.reshape(1,-1).flatten()
        self.hashed_s   = None # We dont want to reuse the hased phi because phi function is changed!
    def hashState(self,s,):
        #returns a unique idea by calculating the enumerated number corresponding to a state
        # it first translate the state into a binState (bin number corresponding to each dimension)
        # it then map the binstate to a an integer
        ds = self.binState(s)
        return vec2id(s,self.bins_per_dim)
    def setBinsPerDimension(self,domain,discretization):
        # Set the number of bins for each dimension of the domain (continous spaces will be slices using the discritization parameter)
        self.bins_per_dim = zeros(domain.state_space_dims,uint16)
        for d in arange(domain.state_space_dims):
             if d in domain.continous_dims:
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
        bs  = empty(len(s),'uint16')
        for d in arange(self.domain.state_space_dims):
            bs[d] = binNumber(s[d],self.bins_per_dim[d],self.domain.statespace_limits[d,:])
        return bs
    def printAll(self):
        printClass(self)
    def bestActions(self,s):
    # Given a state returns the best action possibles at that state
        Qs, A = self.Qs(s)
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
#        for dim in self.domain.continous_dims:
#                ds[dim] = closestDiscretization(ds[dim],self.discretization,self.domain.statespace_limits[dim][:]) 
#        return ds
    def activeInitialFeatures(self,s):
        #return the index of active initial features based on bins on each dimensions
        bs          = self.binState(s)
        shifts      = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index       = bs+shifts
        return      index.astype('uint32')
    
    