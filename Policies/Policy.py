## \file Policy.py
######################################################
# \author Developed by Alborz Geramiard Oct 30th 2012 at MIT 
######################################################
from Tools import *
from Representations import *

## The Policy determines the action that an \ref Agents.Agent.Agent "Agent" will take given its \ref Representations.Representation.Representation "Representation".
# 
# The Agent learns about the \ref Domains.Domain.Domain "Domain" as the two interact. Each step, the Agent passes information about its current state and information
# revelent to that state to the %Policy. The %Policy uses this information to decide what action the Agent should perform next. \n
#
# The \c %Policy class is a superclass that provides the basic framework for all policiess. It provides the methods and attributes
# that allow child classes to interact with the \c Agent and \c Represemtatopm classes within the RL-Python library. \n
# All new policty implimentations should inherit from \c %Policy.

class Policy(object):
	## Link to the representation
    representation = None 
	## In Debug Mode?
    DEBUG          = False
	
	## Initializes the \c %Policy object. See code
	# \ref Policy_init "Here".
	
	# [init code]
    def __init__(self,representation,logger):
        self.representation = representation
		## An object to record the print outs in a file
        self.logger         = logger
	# [init code]
	
	
	## \b ABSTRACT \b METHOD: Select an action given a state. See code
	# \ref Policy_pi "Here".
	
	# [pi code]
    def pi(self,s):
       abstract 
	# [pi code]
	
	
	## \b ABSTRACT \b METHOD: Turn exploration off. See code
	# \ref Policy_turnOffExploration "Here".
	
	# [turnOffExploration code]
    def turnOffExploration(self):
        pass
	# [turnOffExploration code]
	
	
	## \b ABSTRACT \b METHOD: Turn exploration on. See code
	# \ref Policy_turnOnExploration "Here".
	
	# [turnOnExploration code]
    def turnOnExploration(self):
        pass
	# [turnOnExploration code]
	
	
	## Prints class information. See code
	# \ref Policy_printAll "Here".
	
    # [printAll code]
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
	# [printAll code]
    def collectSamples(self, samples):
    		# Return matrices of S,A,NS,R,T where each row of each matrix is a sample by following the current policy
    	 	domain = self.representation.domain
    	 	S 	= empty((samples,self.representation.domain.state_space_dims),dtype = type(domain.s0()))
		A   = empty((samples,1),dtype='uint16')
		NS	= S.copy()
		T 	= A.copy()
		R 	= empty((samples,1))
    		
    		sample 		= 0
    		eps_length 	= 0
    		terminal 	= True # So the first sample forces initialization of s and a
    		while sample < samples:
			if terminal or eps_length > self.representation.domain.episodeCap:
				s = domain.s0()
				a = self.pi(s)
			
			#Transition
			r,ns,terminal = domain.step(s,a)
			#Collect Samples
			S[sample] 	= s
			A[sample] 	= a
			NS[sample]	= ns
			T[sample]	= terminal
			R[sample]	= r
			
			sample += 1
			eps_length += 1
			s = ns
			a = self.pi(s)
			
		return S,A,NS,R,T
    			
    			
    	
    	
    	
    	