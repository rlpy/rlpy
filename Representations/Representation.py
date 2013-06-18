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
    discretization  = 0     #Number of buckets used for discretization for each continuous dimension  
    buckets_per_dim = None  #Number of possible states per dimension [1-by-dim] 
    def __init__(self,domain,discretization = 20):
        # See if the child has set important attributes  
        for v in ['features_num']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.domain = domain
        self.theta  = zeros(self.features_num*self.domain.actions_num) 
        self.discretization = discretization
        #Calculate Buckets per Dimension
        self.buckets_per_dim = zeros(self.domain.state_space_dims,uint8)
        for d in arange(self.domain.state_space_dims):
             if d in self.domain.continous_dims:
                 self.buckets_per_dim[d] = self.discretization
             else:
                 self.buckets_per_dim[d] = self.domain.statespace_limits[d,1] - self.domain.statespace_limits[d,0]+1
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
            return 0
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
    def discretized(self,s):
        # If domain has continuous dimensions, this function returns the closest state based on the discretization otherwise it returns the same state 
        for dim in self.domain.continous_dims:
                s[dim] = closestDiscretization(s[dim],self.discretization,self.domain.statespace_limits[dim][:]) 
        return s
    def addNewWeight(self):
        # Add a new 0 weight corresponding to the new added feature for all actions.
        x               = self.theta.reshape(self.domain.actions_num,-1) # -1 means figure the other dimension yourself
        x               = hstack((x,zeros((self.domain.actions_num,1))))
        self.theta      = x.reshape(1,-1).flatten()
    def hashState(self,s,):
        #returns a unique idea by calculating the enumerated number corresponding to a state
        # I use a recursive calculation to save time by looping once backward on the array = O(n)
        return vec2id(s,self.buckets_per_dim)
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
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
            raw_input()
        return A[ind]
