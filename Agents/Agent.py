######################################################
# Developed by Alborz Geramiard Oct 25th 2012 at MIT #
######################################################
from Tools import *
from Representations import *
class Agent(object):
    representation = None # Link to the representation object 
    domain = None         # Link to the domain object
    policy = None         # Link to the policy object
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

        