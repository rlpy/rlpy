##\file
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
	
    def __init__(self,representation,logger):
        self.representation = representation
		## An object to record the print outs in a file
        self.logger         = logger
	
	## \b ABSTRACT \b METHOD: Select an action given a state
    def pi(self,s):
       abstract 
	## \b ABSTRACT \b METHOD: Turn exploration off
    def turnOffExploration(self):
        pass
	## \b ABSTRACT \b METHOD: Turn exploration on
    def turnOnExploration(self):
        pass
	## Prints class information
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
    