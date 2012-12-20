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
    logger              = None          #A simple objects that record the prints in a file
    def __init__(self,representation,policy,domain,logger):
        self.representation = representation
        self.policy = policy
        self.domain = domain
        self.logger = logger
        if self.logger:
            self.logger.line()
            self.logger.log("Agent:\t\t"+str(className(self)))
            self.logger.log("Policy:\t\t"+str(className(self.policy)))    
    def learn(self,s,a,r,ns,na,terminal):
        abstract
    def printAll(self):
        printClass(self)
    def checkPerformance(self):
        # This function should not be here. This is just for debugging and getting insight into the performance evolution
        # Set Exploration to zero and sample one episode from the domain
        eps_length  = 0
        eps_return  = 0
        eps_term    = 0
        self.policy.turnOffExploration()
        s           = self.domain.s0()
        terminal    = False

        while not eps_term and eps_length < self.domain.episodeCap:
            a               = self.policy.pi(s)
            r,ns,eps_term   = self.domain.step(s, a)
            s               = ns
            eps_return     += r
            eps_length     += 1
        self.policy.turnOnExploration()
        return eps_return, eps_length, eps_term
    