######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
# This object encodes fixed policies for some of the domains.
# 1. InvertedPendulum
from Policy import *
class FixedPolicy(Policy):
    supportedDomains = ['Pendulum_InvertedBalance']
    def pi(self,s):
        if not className(self.representation.domain) in self.supportedDomains:
            print "ERROR: There is no fixed policy defined for %s" % className(self.representation.domain)
            return None
        if className(self.representation.domain) == 'Pendulum_InvertedBalance':
            # Fixed policy rotate the pendulum in the opposite direction of the thetadot
            theta, thetadot = s
            if thetadot > 0:
                return 2
            else:
                return 0
        
        
        
        