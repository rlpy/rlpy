######################################################
# Developed by Alborz Geramiard Nov 21th 2012 at MIT #
######################################################
from Policy import *
class UniformRandom(Policy):
    def pi(self,s):
        A = self.representation.domain.possibleActions(s)
        return randSet(A)
         
        
        
        