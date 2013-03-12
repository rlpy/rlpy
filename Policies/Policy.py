## \file Policy.py
######################################################
# \author Developed by Alborz Geramiard Oct 30th 2012 at MIT 
######################################################
from Tools import *
from Representations import *

## The \c %Policy class controls the  
# 
# \c %Policy provides the basic framework for Agents to interact with Policies and for
# Policies to interact with Representations.
# All new policy implimentations should inherit from \c %Policy.
#
# A policy determines the action that an Agent will take given its representaion.

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
    