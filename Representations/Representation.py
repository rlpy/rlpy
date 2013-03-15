## \file Representation.py
######################################################
# \author Developed by Alborz Geramiard Oct 25th 2012 at MIT 
######################################################

from Tools import *

## The Representation is the \ref Agents.Agent.Agent "Agent"'s model of the \ref Domains.Domain.Domain "Domain".
# 
# As the Agent interacts with the Domain, it receives observations. The Agent passes these observations to its
# %Representation which extracts and stores the useful information by updating its iternal model apporpriately. 
# The %Representation then uses this updated model and the current state of the Agent supplied by the Domain to provide the Agent with
# any information it needs. The Agent then uses this information and its \ref Policies.Policy.Policy "Policy" to select an action to perform. \n
#
# The \c %Representation class is a superclass that provides the basic framework for all representations. It provides the methods and attributes
# that allow child classes to interact with the \c Agent and \c Domain classes within the RL-Python library. \n
# All new representation implimentations should inherit from \c %Representation.
# \note It is assumed that the Linear Function approximator family of representations is being used.
class Representation(object):
	## In Debug Mode?
	DEBUG		   = 0
	## Linear Weights
	theta		   = None  
	## Link to the domain object 
	domain		  = None  
	## Number of features
	features_num	= None  
	## Number of bins used for discretization for each continuous dimension
	discretization  = 0	 
	## Number of possible states per dimension [1-by-dim]	
	bins_per_dim	= None  
	## Width of bins in each dimension
	binWidth_per_dim= None  
	## Number of aggregated states based on the discretization. If the represenation is adaptive set it to the best resolution possible	
	agg_states_num  = None  
	## Object for capturing output text in a file	
	logger = None	   
	
	## Initializes the \c %Representation object. See code
	# \ref Representation_init "Here".
	
	# [init code]
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
		self.logger.log("Features per action:\t%d" % self.features_num)
		if len(self.domain.continuous_dims): 
			self.logger.log("Discretization:\t\t%d"% self.discretization)
			self.logger.log("Starting Features:\t%d"% self.features_num)
			self.logger.log("Aggregated States:\t%d"% self.agg_states_num)
	# [init code]
	
	
	## Returns the value of a state. See code
	# \ref Representation_V "Here".
	
	# [V code]
	def V(self,s, phi_s = None):
		if phi_s is None: phi_s = self.phi(s)
		AllQs,A   = self.Qs(s,phi_s)
		if len(A):
			return max(AllQs)
		else:
			return 0 #Return 0 value when no action is possible
	# [V code]
	
	
	## Returns two arrays
	# Q: array of Q(s,a)
	# A: Corresponding array of action numbers
	# If phi_s is given, it uses that to speed up the process. See code
	# \ref Representation_Qs "Here".
	
	# [Qs code]
	def Qs(self,s, phi_s = None):
		A = self.domain.possibleActions(s)
		if phi_s is None: phi_s   = self.phi(s)
		return array([self.Q(s,a,phi_s) for a in A]), A   
	# [Qs code]
	
	
	##Returns the state-action value. See code
	# \ref Representation_Q "Here".
	
	# [Q code]
	def Q(self,s,a,phi_s = None):
		
		if len(self.theta) > 0:
			return dot(self.phi_sa(s,a, phi_s),self.theta)
		else:
			return 0.0
	# [Q code]
	
	
	##Returns the phi(s). See code
	# \ref Representation_phi "Here".
	
	# [phi code]
	def phi(self,s):
		if self.domain.isTerminal(s) or self.features_num == 0:
			return zeros(self.features_num,'bool')
		else:
			return self.phi_nonTerminal(s)
	# [phi code]
	
	
	## Returns the feature vector corresponding to s,a (we use copy paste technique (Lagoudakis & Parr 2003)
	#If phi_s is passed it is used to avoid phi_s calculation. See code
	# \ref Representation_phi_sa "Here".
	
	# [phi_sa code]
	def phi_sa(self,s,a, phi_s = None):
		if phi_s is None: phi_s = self.phi(s)
		phi_sa = zeros(self.features_num*self.domain.actions_num, dtype=phi_s.dtype)
		ind_a = arange(a*self.features_num,(a+1)*self.features_num)
		phi_sa[ind_a] = phi_s
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
	
	
	## Add a new 0 weight corresponding to the new added feature for all actions.. See code
	# \ref Representation_addNewWeight "Here".
	
	# [addNewWeight code]
	def addNewWeight(self):
		self.theta	  = addNewElementForAllActions(self.theta,self.domain.actions_num)
	# [addNewWeight code]
	
	
	## returns a unique id by calculating the enumerated number corresponding to a state
	# it first translates the state into a binState (bin number corresponding to each dimension)
	# it then maps the binstate to an integer. See code
	# \ref Representation_hashState "Here".
	
	# [hashState code]
	def hashState(self,s,):
		ds = self.binState(s)
		#self.logger.log(str(s)+"=>"+str(ds))
		return vec2id(ds,self.bins_per_dim)
	# [hashState code]
	
	
	## Set the number of bins for each dimension of the domain (continuous spaces will be slices using the discritization parameter). See code
	# \ref Representation_setBinsPerDimension "Here".
	
	# [setBinsPerDimension code]
	def setBinsPerDimension(self,domain,discretization):
		self.bins_per_dim	   = zeros(domain.state_space_dims,uint16)
		self.binWidth_per_dim   = zeros(domain.state_space_dims)
		for d in arange(domain.state_space_dims):
			 if d in domain.continuous_dims:
				 self.bins_per_dim[d] = discretization
			 else:
				 self.bins_per_dim[d] = domain.statespace_limits[d,1] - domain.statespace_limits[d,0]
			 self.binWidth_per_dim[d] = (domain.statespace_limits[d,1] - domain.statespace_limits[d,0])/(self.bins_per_dim[d]*1.)
	# [setBinsPerDimension code]
	
	
	## Given a state it returns a vector with the same dimensionality of s
	# each element of the returned valued is the zero-indexed bin number corresponding to s
	# This function accepts scalar inputs when the domain has 1 dimension 
	# CompactBinary version exclude feature activation for the negative case of binary features.
	# For example if the light is off, no feature corresponds to this case and hence nothing is activated. See code
	# \ref Representation_binState "Here".
	
	# [binState code]
	def binState(self,s):
		if isinstance(s,int): s = [s]
		assert(len(s) == len(self.domain.statespace_limits[:,0]))
		bs  = empty(len(s),'uint16')
		for d in arange(self.domain.state_space_dims):
			bs[d] = binNumber(s[d],self.bins_per_dim[d],self.domain.statespace_limits[d,:])
		return bs
	# [binState code]
	
	
	## Prints the class information. See code
	# \ref Representation_printAll "Here".
	
	# [printAll code]
	def printAll(self):
		printClass(self)
	# [printAll code]
	
	
	## Given a state returns the best action possibles at that state
	# If phi_s is given it is used to speed up. See code
	# \ref Representation_bestActions "Here".
	
	# [bestActions code]
	def bestActions(self,s, phi_s = None):
		Qs, A = self.Qs(s,phi_s)
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
		return A[ind]
	# [bestActions code]
	
	
#	def discretized(self,s):
#		ds = s.copy()
#		for dim in self.domain.continuous_dims:
#				ds[dim] = closestDiscretization(ds[dim],self.discretization,self.domain.statespace_limits[dim][:]) 
#		return ds


	## return an action among the best actions uniformly randomly. See code
	# \ref Representation_bestAction "Here".
	
	# [bestAction code]
	def bestAction(self,s, phi_s = None):
		bestA = self.bestActions(s,phi_s)
		if len(bestA) > 1:
			return randSet(bestA)
			#return bestA[0]
		else:
			return bestA[0]
	# [bestAction code]
	
	
	##\b ABSTRACT \b METHOD: This is the actual function that each representation should fill
	# \note if state is terminal the feature vector is always zero! See code
	# \ref Representation_phi_nonTerminal "Here".
	
	# [phi_nonTerminal code]
	def phi_nonTerminal(self,s):
			abstract
	# [phi_nonTerminal code]
	
	
	## return the index of active initial features based on bins on each dimensions. See code
	# \ref Representation_activeInitialFeatures "Here".
	
	# [activeInitialFeatures code]
	def activeInitialFeatures(self,s):
		bs		  = self.binState(s)
		shifts	  = hstack((0, cumsum(self.bins_per_dim)[:-1]))
		index	   = bs+shifts
		return	  index.astype('uint32')
	# [activeInitialFeatures code]
	
	
	## Discovers features and adds it to the representation
	# If it adds any feature it should return True, otherwise False
	# This is a dummy function for representations with no discovery
	# TD_error is a vector of TD-Errors for all samples p-by-1
	# all_phi_s is phi(s) for all s in (s,a,r,s',a') p-by-|dim(phi(s))|
	# data_s is the states p-by-|dim(s)|. See code
	# \ref Representation_batchDiscover "Here".
	
	# [batchDiscover code]
	def batchDiscover(self, td_errors, all_phi_s, data_s):
		return False
	# [batchDiscover code]
	
	
	## Input: 
	# all_phi_s p-by-n [feature vectors]
	# all_actions p-by-1 [set of actions corresponding to each feature
	# Optional) If phi_s_a has been built for all actions pass it for speed boost
	# output:
	# returns all_phi_s_a p-by-na. See code
	# \ref Representation_batchPhi_s_a "Here".
	
	# [batchPhi_s_a code]
	def batchPhi_s_a(self,all_phi_s, all_actions, all_phi_s_a = None, use_sparse = False):		
		p,n			= all_phi_s.shape
		a_num		= self.domain.actions_num
		phi_s_a 		= zeros((p,n*a_num),dtype = all_phi_s.dtype)					
	
		for i in arange(a_num):
			rows = where(all_actions==i)[0]
			if len(rows): phi_s_a[rows,i*n:(i+1)*n] = all_phi_s[rows,:]
		return phi_s_a
	# [batchPhi_s_a code]
	
	
		# closed on 2/27/2013: Effort to sparsify things
		#=====================================================================
#		if all_phi_s_a == None: 
#			if use_sparse:
#				all_phi_s_a = sp.kron(eye(a_num,a_num, dtype = uint8),all_phi_s) #all_phi_s_a will be ap-by-an
#			else:
#				all_phi_s_a = kron(eye(a_num,a_num, dtype = bool),all_phi_s)		 #all_phi_s_a will be ap-by-an
#		
#		if sp.issparse(all_phi_s_a):
#				all_phi_s_a = all_phi_s_a.todense()
#				
#		# Based on Josh's Idea
#		# set_printoptions(threshold=sys.maxint, precision=2, suppress=True, linewidth=inf)
#		M = all_phi_s_a
#		M = M.reshape((a_num,-1))
#		A = all_actions.T
#		A = kron(A,ones((1,n*a_num,),dtype=integer))[0] # <<< SPARSIFY if you have time
#		M = M[A,arange(len(A)),:]
#		M = M.reshape(-1)
#		return M.reshape((p,-1))
		
		# Below is the first matrix attemp which is slightly slower than the above method based on 1 run of PST performance	
#	   use_sparse = 1
#		action_slice	= zeros((a_num,p),dtype= bool)
#		action_slice[all_actions,xrange(p)] = 1
		# Build a matrix where 1 appears in each column corresponding to the action number
		# all_actions = [1 0 1] with 2 actions and 3 samples
		# build: 
		# 0 1 0
		# 1 0 1
		#now expand each 1 into size of the features (i.e. n)
#		all_phi_s_a = all_phi_s_a.reshape((a_num,-1))
#		
#		if use_sparse:
#			action_slice = sp.kron(sp.csr_matrix(action_slice),ones((1,n*a_num)),'coo')
#			action_slice = action_slice.todense()
#			phi_s_a = all_phi_s_a.T[action_slice.T==1]
#		else:
#			action_slice = kron(action_slice,ones((1,n*a_num)))
#			phi_s_a = all_phi_s_a.T[action_slice.T==1]
#		
#		# with n = 2, and a = 2 we will have:
#		# 0 0 0 0 1 1 1 1 0 0 0 0
#		# 1 1 1 1 0 0 0 0 1 1 1 1
#		# now we can select the feature values
#		#phi_s_a = all_phi_s_a.T[action_slice.todense().T==1]
#		phi_s_a = phi_s_a.reshape((p,-1))
#		return phi_s_a


	## Returns the best-action and phi_s_a corresponding to the states
	# inputs:
	# 1: all-s: p-by-dim(s)
	# 2: all_phi_s: p-by-|phi(s))|
	# 3: Optional) If action mask is available it can be passed to boost the calculation
	# outputs:
	# best_action: p-by-1
	# phi_s_a: p-by-|phi(s,a)|
	# action_mask
	
	# Algorithm:
	# 1. Calculate the phi_s_a for all actions and given s for each row
	# 2. Multiply theta to the corresponding phi_s_a
	# 3. Rearrange the matrix to have in each row all values corresponding to possible actions
	# 4. Maskout irrelevant actions
	# 5. find the max index in each row
	# 6. return the action and corresponding_phi_s_a
	
	# make a mask for the invalid_actions
	# build a matrix p-by-a where in each row the missing action is 1
	# Example: 2 actions, 3 states
	# possibleActions(s1) = 0
	# possibleActions(s2) = 1
	# possibleActions(s3) = 0,1
	# output =>  0 1
	#			1 0
	#			0 0 
	
	
	## DESCRIPTION. See code
	# \ref Representation_batchBestAction "Here".
	
	# [batchBestAction code]
	def batchBestAction(self, all_s, all_phi_s, action_mask = None, useSparse = True):
		p,n	 = all_phi_s.shape
		a_num   = self.domain.actions_num
		
		if action_mask == None:
			action_mask = ones((p,a_num))
			for i,s in enumerate(all_s):
				action_mask[i,self.domain.possibleActions(s)] = 0 
		
		a_num	   = self.domain.actions_num
		if useSparse:
				all_phi_s_a = sp.kron(eye(a_num,a_num),all_phi_s)	 #all_phi_s_a will be ap-by-an
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
	def Q_oneStepLookAhead(self,s,a, ns_samples):
		# Returns the Q(s,a) by performing one step look-ahead (e.g. Line 8 of Figure 4.3 in Sutton and Barto 1998) using the domain.
		# As a result this function should not be called in any RL algorithms unless the underlying domain is approximation of the true model
		# If domain does not have expectedStep function, it uses <ns_samples> samples to estimate the one_step look-ahead  
		gamma 	= self.domain.gamma 
		if hasFunction(self.domain,'expectedStep'):
			p,r,ns,t	= self.domain.expectedStep(s,a)
			Q	 	= 0
			for j in arange(len(p)):
				Q += p[j]*(r[j] + gamma*self.V(ns[j,:]))
			Q = Q[0]
		else:
			next_states,rewards = self.domain.sampleStep(s,a,ns_samples)
			Q = mean([rewards[i] + gamma*self.V(next_states[i,:]) for i in arange(ns_samples)])
		return Q			
	def Qs_oneStepLookAhead(self,s, ns_samples):
		# Returns Q(s,a) for all possible actions by performing one step look-ahead (e.g. Line 8 of Figure 4.3 in Sutton and Barto 1998) using the domain.
		# As a result this function should not be called in any RL algorithms unless the underlying domain is approximation of the true model
		# If domain does not have expectedStep function, it uses <ns_samples> samples to estimate the one_step look-ahead  
		# It also returns all posssible actions as the second output
		actions = self.domain.possibleActions(s)
		Qs 		= array([self.Q_oneStepLookAhead(s, a, ns_samples) for a in actions]) 
		return Qs, actions
	def V_oneStepLookAhead(self,s,ns_samples):
		# Returns V(s) and the corresponding action by performing one step look-ahead (e.g. Line 6 of Figure 4.5 in Sutton and Barto 1998) using the domain.
		# As a result this function should not be called in any RL algorithms unless the underlying domain is approximation of the true model
		# If domain does not have expectedStep function, it uses <ns_samples> samples to estimate the one_step look-ahead  
		
		Qs, actions 	= self.Qs_oneStepLookAhead(s,ns_samples)
		a_ind   		= argmax(Qs)
		return Qs[a_ind],actions[a_ind]		 
			
			
			
			