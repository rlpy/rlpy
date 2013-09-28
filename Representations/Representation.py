#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## \file Representation.py
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT
######################################################

from Tools import *
## The Representation is the \ref Agents.Agent.Agent "Agent"'s model of the \ref Domains.Domain.Domain "Domain".
#
# As the Agent interacts with the Domain, it receives state updates. The Agent passes these associated state-action / state-action pairings
# along with the associated reward for each timestep to its
# %Representation, which is responsible for maintaining the value function for each state, usually in some
# lower-dimensional feature space.
# Agents can later query the %Representation for the value of being in a state V(s) or the value of taking an action
# in a particular state ( known as the Q-function, Q(s,a) ).
#
# The \c %Representation class is a superclass that provides the basic framework for all representations. It provides the methods and attributes
# that allow child classes to interact with the \c Agent and \c Domain classes within the RLPy library. \n
# All new representation implimentations should inherit from \c %Representation.
# \note It is assumed that the Linear Function approximator family of representations is being used.

class Representation(object):
    ## In Debug Mode?
    ## \cond DEV
    DEBUG          = 0
    ## \endcond
    ## A numpy array of the Linear Weights
    theta          = None
    ## The \ref Domains.Domain.Domain "Domain" that the %Representation is modeling
    domain        = None
    ## Number of features
    features_num    = 0
    ## Number of bins used for discretization of each continuous dimension
    discretization  = 20
    ## Number of possible states per dimension [1-by-dim]
    bins_per_dim    = 0
    ## Width of bins in each dimension
    binWidth_per_dim= 0
    ## Number of aggregated states based on the discretization. If the represenation is adaptive, set to the best resolution possible
    agg_states_num  = 0
    ## A simple object that records the prints in a file
    logger = None

    ## A boolan stating that if the representation is dynamic meaning the size of features is going to change (default value = False).
    isDynamic = False
    ## A dictionary used to cache expected step. Used for planning algorithms
    expectedStepCached=None
    ## Initializes the \c %Representation object. See code
    # \ref Representation_init "Here".

    # [init code]
    def __init__(self,domain,logger,discretization = 20):
        # See if the child has set important attributes
        for v in ['features_num']:
            if getattr(self,v) == None:
                raise Exception('Missed domain initialization of '+ v)
        self.expectedStepCached = {}
        self.setBinsPerDimension(domain,discretization)
        self.domain = domain
        self.discretization = discretization
        try:
            self.theta  = zeros(self.features_num*self.domain.actions_num)
        except MemoryError as m:
            print("Unable to allocate weights of size: %d\n" % self.features_num*self.domain.actions_num)
            raise m

        self._phi_sa_cache = empty((self.domain.actions_num, self.features_num))
        self._arange_cache = arange(self.features_num)
        self.agg_states_num = prod(self.bins_per_dim.astype('uint64'))
        self.logger = logger
        self.logger.line()
        self.logger.log("Representation:\t\t%s" % className(self))
        self.logger.log("Features per action:\t%d" % self.features_num)
        if len(self.domain.continuous_dims):
            self.logger.log("Discretization:\t\t%d"% self.discretization)
            self.logger.log("Starting Features:\t%d"% self.features_num)
            self.logger.log("Aggregated States:\t%d"% self.agg_states_num)
    # [init code]


    ## Returns the value of a state. See code
    # \ref Representation_V "Here".

    # [V code]
    def V(self, s, terminal, p_actions, phi_s = None):
        if phi_s is None: phi_s = self.phi(s, terminal)
        AllQs   = self.Qs(s, terminal, phi_s)
        if len(p_actions):
            return max(AllQs[p_actions])
        else:
            return 0 #Return 0 value when no action is possible
    # [V code]


    ## Returns an array of actions available at a state and their associated values.
    # If phi_s is given, the function uses it to speed up the process. See code
    # \ref Representation_Qs "Here".
    # @param s The state to examine.
    # @param phi_s The feature vector evaluated at state (s).
    # @return [Q,A] \n
    # \b Q: an array of Q(s,a), the values of each action. \n
    # \b A: the corresponding array of action numbers

    # [Qs code]
    def Qs(self,s, terminal, phi_s = None):

        if phi_s is None: phi_s   = self.phi(s, terminal)
        if len(phi_s) == 0:
            return np.zeros((self.domain.actions_num))
        theta_prime = self.theta.reshape(-1, self.features_num)
        if self._phi_sa_cache.shape != (self.domain.actions_num, self.features_num):
            self._phi_sa_cache =  empty((self.domain.actions_num, self.features_num))
        Q = multiply(theta_prime, phi_s, out=self._phi_sa_cache).sum(axis=1) # stacks phi_s in cache
        return Q
    # [Qs code]


    ## Returns the state-action value. See code
    # \ref Representation_Q "Here".
    # @param s The state to examine.
    # @param a The action to examine.
    # @param phi_s The feature vector evaluated at state (s)
    # @return The value of the action in that state.

    # [Q code]
    def Q(self, s, terminal, a, phi_s = None):
        if len(self.theta) > 0:
            phi_sa, i, j = self.phi_sa(s, terminal, a, phi_s, snippet=True)
            return dot(phi_sa,self.theta[i:j])
        else:
            return 0.0
    # [Q code]


    ## Returns phi_nonTerminal(s) for a given representation, or a zero feature vector in a terminal state.
    # See code
    # \ref Representation_phi "Here".
    # @param s The given state
    # @returns Phi. Format is [[]].

    # [phi code]
    def phi(self,s, terminal):
        if terminal or self.features_num == 0:
            return zeros(self.features_num,'bool')
        else:
            return self.phi_nonTerminal(s)
    # [phi code]


    ## Returns the feature vector corresponding to a given state and action.
    # We use the copy paste technique (Lagoudakis & Parr 2003)
    # if phi_s is passed it is used to avoid phi_s calculation. See code
    # \ref Representation_phi_sa "Here".
    # @param s The given state
    # @param a The given action
    # @param phi_s The feature vector evaluated at state (s)
    # @return The associated feature vector.

    # [phi_sa code]
    def phi_sa(self, s, terminal, a, phi_s = None, snippet=False):
        if phi_s is None: phi_s = self.phi(s, terminal)
        if snippet is True:
            return phi_s, a*self.features_num, (a+1) * self.features_num

        phi_sa = zeros((self.features_num*self.domain.actions_num), dtype=phi_s.dtype)
        if self.features_num == 0:
            return phi_sa
        if len(self._arange_cache) != self.features_num:
            self._arange_cache = arange(a * self.features_num, (a+1) * self.features_num)
        else:
            self._arange_cache += a*self.features_num - self._arange_cache[0]
        phi_sa[self._arange_cache] = phi_s
        ##Slower alternatives
        ##Alternative 1: Set only non_zeros (Very close on running time with the current solution. In fact it is sometimes better)
        #nnz_ind = phi_s.nonzero()
        #phi_sa[nnz_ind+a*self.features_num] = phi_s[nnz_ind]
        ##Alternative 2: Use of Kron
        #A = zeros(self.domain.actions_num)
        #A[a] = 1
        #F_sa = kron(A,F_s)
        return phi_sa
    # [phi_sa code]


    ## Add a new zero weight, corresponding to a newly added feature, to all actions. See code
    # \ref Representation_addNewWeight "Here".

    # [addNewWeight code]
    def addNewWeight(self):
        self.theta    = addNewElementForAllActions(self.theta,self.domain.actions_num)
    # [addNewWeight code]


    ## Returns a unique id for a given state.
    # It operates by calculating the enumerated number corresponding to said state.
    # First it translates the state into a binState (bin number corresponding to each dimension).
    # Then it maps the binstate to an integer. See code
    # \ref Representation_hashState "Here".
    # @param s The given state
    # @returns The unique id of the state.

    # [hashState code]
    def hashState(self,s,):
        ds = self.binState(s)
        #self.logger.log(str(s)+"=>"+str(ds))
        return vec2id(ds,self.bins_per_dim)
    # [hashState code]


    ## Set the number of bins for each dimension of the domain.
    # Continuous spaces will be slices using the discritization parameter. See code
    # \ref Representation_setBinsPerDimension "Here".
    # @param domain The \ref Domains.Domain.Domain "Domain" associated with the %Representation
    # @param discretization The number of slices a continous domain should be sliced into.

    # [setBinsPerDimension code]
    def setBinsPerDimension(self,domain,discretization):
        self.bins_per_dim      = np.zeros(domain.state_space_dims,uint16)
        self.binWidth_per_dim   = np.zeros(domain.state_space_dims)
        for d in arange(domain.state_space_dims):
             if d in domain.continuous_dims:
                 self.bins_per_dim[d] = discretization
             else:
                 self.bins_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]
             self.binWidth_per_dim[d] = (domain.statespace_limits[d,1] - domain.statespace_limits[d,0])/(self.bins_per_dim[d]*1.)
    # [setBinsPerDimension code]

    def binState(self, s):
        """Returns a vector where each element is the zero-indexed bin number corresponding with the given state.
        Note that this vector will have the same dimensionality of the given state.
        Each element of the returned vector is the zero-indexed bin number corresponding with the given state.
        This method is binary compact; the negative case of binary features is excluded from feature activation.
        For example, if the domain has a light and the light is off, no feature will be added. This is because no
        features correspont to a light being off. See code \ref Representation_binState "Here".

        @param s The given state, can be a scalar value if the dimension is one dimensional.
        @return The desired vector
        """
        s = np.atleast_1d(s)
        limits = self.domain.statespace_limits
        assert (np.all(s >= limits[:, 0]))
        assert (np.all(s <= limits[:, 1]))
        width = limits[:, 1] - limits[:, 0]
        diff = s - limits[:, 0]
        bs = (diff * self.bins_per_dim / width).astype("uint32")
        m = bs == self.bins_per_dim
        bs[m] = self.bins_per_dim[m] - 1
        return bs

    ## Returns a list of the best actions at a given state.
    # If phi_s [the feature vector at state (s)]is given, it is used to speed up code by preventing re-computation. See code
    # \ref Representation_bestActions "Here".
    # @param s The given state
    # @param phi_s the feature vector at state (s)
    # @return A list of the best actions at the given state.

    # [bestActions code]
    def bestActions(self,s, terminal, p_actions, phi_s = None):
        Qs = self.Qs(s, terminal, phi_s)
        Qs = Qs[p_actions]
        # Find the index of best actions
        ind   = findElemArray1D(Qs,Qs.max())
        if self.DEBUG:
            self.logger.log('State:' +str(s))
            self.logger.line()
            for i in arange(len(A)):
                self.logger.log('Action %d, Q = %0.3f' % (A[i], Qs[i]))
            self.logger.line()
            self.logger.log('Best: %s, Max: %s' % (str(A[ind]),str(Qs.max())))
            #raw_input()
        return p_actions[ind]
    # [bestActions code]

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return 0

    def post_discover(self, s, terminal, a, td_error, phi_s):
        return 0


    ## Returns the best action at a given state.
    # If there are multiple best actions, this method selects one of them uniformly randomly. See code
    # \ref Representation_bestAction "Here".
    # @param s The given state
    # @param phi_s the feature vector at state (s)
    # @return The best action at the given state.

    # [bestAction code]
    def bestAction(self,s, terminal, p_actions, phi_s = None):
        bestA = self.bestActions(s, terminal, p_actions, phi_s)
        if len(bestA) > 1:
            return randSet(bestA)
            #return bestA[0]
        else:
            return bestA[0]
    # [bestAction code]


    ##\b ABSTRACT \b METHOD: Returns the feature vector evaluated at state (s) for non-terminal states; see function phi(s) for the general case.
    # See code \ref Representation_phi_nonTerminal "Here".
    # \note if state is terminal the feature vector is always zero!
    # @param s The given state

    # [phi_nonTerminal code]
    def phi_nonTerminal(self,s):
            abstract
    # [phi_nonTerminal code]


    ## returns the index of active initial features based on bins in each dimensions. See code
    # \ref Representation_activeInitialFeatures "Here".
    # @param s The given state
    # @return The desired index

    # [activeInitialFeatures code]
    def activeInitialFeatures(self,s):
        bs        = self.binState(s)
        shifts    = hstack((0, cumsum(self.bins_per_dim)[:-1]))
        index      = bs+shifts
        return    index.astype('uint32')
    # [activeInitialFeatures code]


    # \b ABSTRACT \b METHOD: Discovers features from a collection of data ('p' samples) and adds them to the representation.
    # Representations that do not have discovery do not have to overwrite this method. See code
    # \ref Representation_batchDiscover "Here".
    # @param td_errors A vector of TD (Temporal Difference)-Errors for all 'p' samples [p-by-1 vector]
    # @param all_phi_s Feature vector Phi evaluated at state (s), p-by-|dim(phi(s))| (i.e. there are p rows, each containing the feature vector phi_s for a single state s)
    # @param data_s The states themselves, p-by-|dim(s)|
    # @return A boolean stating whether the method added a feature or not.

    # [batchDiscover code]
    # def batchDiscover(self, td_errors, all_phi_s, data_s):
    #   return False
    # [batchDiscover code]


    ## Build the feature vector for a series of state-action pairs (s,a) using the copy-paste method. See code
    # \ref Representation_batchPhi_s_a "Here".
    # @param all_phi_s The feature vectors. p-by-n, where p is the number of s-a pairs (indexed by row), and n is the number of features.
    # @param all_actions The set of actions corresponding to each feature. p-by-1, where p is the number of states included in this batch.
    # @param all_phi_s_a fOptional: Feature vector for a series of state-action pairs (s,a) using the copy-paste method.
    # If phi_s_a has already been built for all actions, pass it for speed boost.
    # @param use_sparse Determines whether or not to use sparse matrix libraries provided with numpy
    # @return all_phi_s_a (of dimension p x (s_a) )

    # [batchPhi_s_a code]
    def batchPhi_s_a(self,all_phi_s, all_actions, all_phi_s_a = None, use_sparse = False):
        p,n         = all_phi_s.shape
        a_num       = self.domain.actions_num
        if use_sparse:
            phi_s_a         = sp.lil_matrix((p,n*a_num),dtype = all_phi_s.dtype)
        else:
            phi_s_a         = zeros((p,n*a_num),dtype = all_phi_s.dtype)

        for i in arange(a_num):
            rows = where(all_actions==i)[0]
            if len(rows): phi_s_a[rows,i*n:(i+1)*n] = all_phi_s[rows,:]
        return phi_s_a
    # [batchPhi_s_a code]

    # [batchBestAction code]
    def batchBestAction(self, all_s, all_phi_s, action_mask = None, useSparse = True):
        p,n  = all_phi_s.shape
        a_num   = self.domain.actions_num

        if action_mask == None:
            action_mask = ones((p,a_num))
            for i,s in enumerate(all_s):
                action_mask[i,self.domain.possibleActions(s)] = 0

        a_num      = self.domain.actions_num
        if useSparse:
                all_phi_s_a = sp.kron(eye(a_num,a_num),all_phi_s)    #all_phi_s_a will be ap-by-an
                all_q_s_a   = all_phi_s_a*self.theta.reshape(-1,1) #ap-by-1
        else:
                all_phi_s_a = kron(eye(a_num,a_num),all_phi_s) #all_phi_s_a will be ap-by-an
                all_q_s_a   = dot(all_phi_s_a,self.theta.T) #ap-by-1
        all_q_s_a   = all_q_s_a.reshape((a_num,-1)).T  #a-by-p
        all_q_s_a   = ma.masked_array(all_q_s_a, mask=action_mask)
        best_action = argmax(all_q_s_a,axis=1)

        # Calculate the corresponding phi_s_a
        phi_s_a = self.batchPhi_s_a(all_phi_s, best_action, all_phi_s_a, useSparse)
        return best_action, phi_s_a, action_mask
    # [batchBestAction code]


    ## \b ABSTRACT \b METHOD: Return the data type for features. See code
    # \ref Representation_featureType "Here".

    # [featureType code]
    def featureType(self):
        abstract
    # [featureType code]


    ## Returns the state action value, Q(s,a), by performing one step look-ahead on the domain.
    # An example of how this function works can be found on Line 8 of Figure 4.3 in Sutton and Barto 1998.
    # If the domain does not have expectedStep function, this function uses ns_samples samples to estimate the one_step look-ahead. See code
    # If policy is passed (used in the policy evaluation), it is used to generate the action for the next state. Otherwise the best action is selected.
    # \ref Representation_Q_oneStepLookAhead "Here".
    # \note This function should not be called in any RL algorithms unless the underlying domain is an approximation of the true model
    # @param s The given state
    # @param a The given action
    # @param ns_samples The number of samples used to estimate the one_step look-aghead.
    # @param policy The optional parameter to decide about the action to be selected in the next state when estimating the one_step look-aghead. If not set the best action will be selected.
    # @return \b Q: The state-action value.

    # [Q_oneStepLookAhead code]
    def Q_oneStepLookAhead(self,s,a, ns_samples, policy = None):
        # Hash new state for the incremental tabular case
        self.continuous_state_starting_samples = 10
        if hasFunction(self,'addState'): self.addState(s)

        gamma   = self.domain.gamma
        if hasFunction(self.domain,'expectedStep'):
            p,r,ns,t,p_actions    = self.domain.expectedStep(s,a)
            Q       = 0
            for j in arange(len(p)):
                if policy == None:
                    Q += p[j,0]*(r[j,0] + gamma*self.V(ns[j,:], t[j,:], p_actions[j]))
                else: 
                    # For some domains such as blocks world, you may want to apply bellman backup to impossible states which may not have any possible actions.
                    # This if statement makes sure that there exist at least one action in the next state so the bellman backup with the fixed policy is valid
                    if len(self.domain.possibleActions(ns[j,:])):
                        na = policy.pi(ns[j,:], t[j,:], self.domain.possibleActions(ns[j,:]))
                        Q += p[j,0]*(r[j,0] + gamma*self.Q(ns[j,:],t[j,:], na))
        else:    
            # See if they are in cache:
            key = tuple(hstack((s,[a])))
            cacheHit     = self.expectedStepCached.get(key)
            if cacheHit is None:
#               # Not found in cache => Calculate and store in cache
                # If continuous domain, sample <continuous_state_starting_samples> points within each discritized grid and sample <ns_samples>/<continuous_state_starting_samples> for each starting state.
                # Otherwise take <ns_samples> for the state.

                #First put s in the middle of the grid:
                #shout(self,s)
                s = self.stateInTheMiddleOfGrid(s)
                #print "After:", shout(self,s)
                if len(self.domain.continuous_dims):
                    next_states = empty((ns_samples,self.domain.state_space_dims))
                    rewards         = empty(ns_samples)
                    ns_samples_ = ns_samples/self.continuous_state_starting_samples # next states per samples initial state
                    for i in arange(self.continuous_state_starting_samples):
                        #sample a random state within the grid corresponding to input s
                        new_s = s.copy()
                        for d in arange(self.domain.state_space_dims):
                            w = self.binWidth_per_dim[d]
                            # Sample each dimension of the new_s within the cell
                            new_s[d] = (random.rand()-.5)*w+s[d]
                            # If the dimension is discrete make make the sampled value to be int
                            if not d in self.domain.continuous_dims:
                                new_s[d] = int(new_s[d])
                        #print new_s
                        ns,r = self.domain.sampleStep(new_s,a,ns_samples_)
                        next_states[i*ns_samples_:(i+1)*ns_samples_,:] = ns
                        rewards[i*ns_samples_:(i+1)*ns_samples_] = r
                else:
                    next_states,rewards = self.domain.sampleStep(s,a,ns_samples)
                self.expectedStepCached[key] = [next_states, rewards]
            else:
                #print "USED CACHED"
                next_states, rewards = cacheHit
            if policy == None:
                Q = mean([rewards[i] + gamma*self.V(next_states[i,:]) for i in arange(ns_samples)])
            else:
                Q = mean([rewards[i] + gamma*self.Q(next_states[i,:],policy.pi(next_states[i,:])) for i in arange(ns_samples)])
        return Q
    # [Q_oneStepLookAhead code]


    ## Returns an array of actions available at a state and their associated values, Qs(s,a), by performing one step look-ahead on the domain.
    # An example of how this function works can be found on Line 8 of Figure 4.3 in Sutton and Barto 1998.
    # If the domain does not have expectedStep function, this function uses ns_samples samples to estimate the one_step look-ahead. See code
    # \ref Representation_Qs_oneStepLookAhead "Here".
    # \note This function should not be called in any RL algorithms unless the underlying domain is an approximation of the true model
    # @param s The given state
    # @param ns_samples The number of samples used to estimate the one_step look-aghead.
    # @param policy The optional parameter to decide about the action to be selected in the next state when estimating the one_step look-aghead.
    # @return [Qs, actions] \n
    # \b Qs: an array of Q(s,a), the values of each action. \n
    # \b actions: the corresponding array of action numbers

    # [Qs_oneStepLookAhead code]
    def Qs_oneStepLookAhead(self,s, ns_samples, policy = None):
        actions = self.domain.possibleActions(s)
        Qs      = array([self.Q_oneStepLookAhead(s, a, ns_samples, policy) for a in actions])
        return Qs, actions
    # [Qs_oneStepLookAhead code]


    ## Returns V(s) by performing one step look-ahead on the domain.
    # An example of how this function works can be found on Line 6 of Figure 4.5 in Sutton and Barto 1998.
    # If the domain does not have expectedStep function, this function uses ns_samples samples to estimate the one_step look-ahead. See code
    # \ref Representation_V_oneStepLookAhead "Here".
    # \note This function should not be called in any RL algorithms unless the underlying domain is an approximation of the true model
    # @param s The given state
    # @param ns_samples The number of samples used to estimate the one_step look-aghead.
    # @return
    # The estimated value = max_a Q(s,a) together with the corresponding action that maximizes the Q function


    # [V_oneStepLookAhead code]
    def V_oneStepLookAhead(self,s,ns_samples):
        Qs, actions     = self.Qs_oneStepLookAhead(s,ns_samples)
        a_ind           = argmax(Qs)
        return Qs[a_ind],actions[a_ind]
    # [V_oneStepLookAhead code]

    ## Returns the state vector correponding to a state_id
    # If dimensions are continuous it returns the state representing the middle of the bin (each dimension is discritized using a parameter into a set of bins)
    # @param s_id The id of the state, often calculated using the state2bin function

    # [stateID2state code]
    def stateID2state(self,s_id):

        #Find the bin number on each dimension
        s   = array(id2vec(s_id,self.bins_per_dim))

        #Find the value corresponding to each bin number
        for d in arange(self.domain.state_space_dims):
            s[d] = bin2state(s[d],self.bins_per_dim[d],self.domain.statespace_limits[d,:])

        if len(self.domain.continuous_dims) == 0:
            s = s.astype(int)
        return s
    # [stateID2state code]

    ## This function returns the state in the middle of the grid that captures the input state.
    # For continuous MDPs this plays a major rule in improving the speed through caching of next samples
    # @param s The given state

    # [stateInTheMiddleOfGrid]
    def stateInTheMiddleOfGrid(self,s):
        s_normalized = s.copy()
        for d in arange(self.domain.state_space_dims):
            s_normalized[d] = closestDiscretization(s[d],self.bins_per_dim[d],self.domain.statespace_limits[d,:])
        return s_normalized
    # [stateInTheMiddleOfGrid]

