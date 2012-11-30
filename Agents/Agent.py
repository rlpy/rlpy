######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from Representations import *
class Agent(object):
    representation      = None # Link to the representation object 
    domain              = None         # Link to the domain object
    policy              = None         # Link to the policy object
    initial_alpha       = 0             #initial learning rate
    alpha               = 0             #Learning rate to be used
    candid_alpha        = 0             #Candid Learning Rate Calculated by [Dabney W. 2012]
    eligibility_trace   = []            #In case lambda parameter in SARSA definition is used
    def __init__(self,representation,policy,domain):
        self.representation = representation
        self.policy = policy
        self.domain = domain
    def learn(self,s,a,r,ns,na,terminal):
        abstract
    def printAll(self):
        printClass(self)
    def printInfo(self):
        print join(["-"]*30)
        print "Agent:\t\t", className(self)
        print "Policy:\t\t", className(self.policy)

        