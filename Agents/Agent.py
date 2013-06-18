######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from Representations import *
class Agent(object):
    name = ''             # Name of the Domain
    possible_actions = [] # Set of possible actions
    representation = None # Link to the representation object 
    domain = None         # Link to the domain object
    policy = None         # Link to the policy object
    pre_s = None          # Previous state         
    pre_a = None          # Previous action
    def __init__(self,representation,policy,domain):
        self.representation = representation
        self.policy = policy
        self.domain = domain
    def learn(self,s,a,r,ns,na):
        abstract
    def printAll(self):
        print className(self)
        print '======================================='
        for property, value in vars(self).iteritems():
            print property, ": ", value

        