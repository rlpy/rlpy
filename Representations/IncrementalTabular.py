######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
from Representation import *
class IncrementalTabular(Representation):
    hash = {}
    def __init__(self,domain,discretization = 20):
        self.features_num = 0
        super(IncrementalTabular,self).__init__(domain,discretization)
    def phi(self,s):
        hash_id = self.hashState(s)
        id  = self.hash.get(hash_id)
        F_s = zeros(self.features_num,'bool')
        if id is not None:
            F_s[id] = 1
        return F_s
    def addState(self,s):
        hash_id = self.hashState(s)
        id  = self.hash.get(hash_id)
        if id is None:
            #New State
            self.features_num += 1
            #New id = feature_num - 1
            id = self.features_num - 1
            self.hash[hash_id] = id
            #Add a new element to weight theta
            self.addNewWeight()

         
        
