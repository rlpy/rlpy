#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Bob Klein
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
# that allow child classes to interact with the \c Agent and \c Represemtatopm classes within the RLPy library. \n
# All new policty implimentations should inherit from \c %Policy.

class Policy(object):
	## The \ref Representations.Representation.Representation "Representation" to be associated with
    representation = None 
	## \cond DEV
    DEBUG          = False
	# \endcond
	
	
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
    			
    			
    	
    	
    	
    	