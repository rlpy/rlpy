######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
from Tools import *
from Representations import *
class Policy(object):
    representation = None #Link to the representation
    DEBUG          = False
    def __init__(self,representation,logger):
        self.representation = representation
        self.logger         = logger
    def pi(self,s):
    # Select an action given a state
       abstract 
    def turnOffExploration(self):
        #Turn exploration off
        abstract
    def turnOnExploration(self):
        #Turn exploration off
        abstract
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value
    