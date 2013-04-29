#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Nov 9th 2012 at MIT #
######################################################
from Representation import *
class IndependentDiscretization(Representation):
    def __init__(self,domain,logger,discretization = 20):
        self.setBinsPerDimension(domain,discretization)
        self.features_num = int(sum(self.bins_per_dim))
        self.maxFeatureIDperDimension = cumsum(self.bins_per_dim)-1
        super(IndependentDiscretization,self).__init__(domain,logger,discretization)
    def phi_nonTerminal(self,s):
        #print "IN PHI:", s,bs, 'shift',shifts,'index:', index, 'Bins:', self.bins_per_dim
        F_s                                 = zeros(self.features_num,'bool')
        F_s[self.activeInitialFeatures(s)]  = 1
        return F_s
    def getDimNumber(self,f):
        # Returns the dimension number corresponding to this feature
        dim     = searchsorted(self.maxFeatureIDperDimension,f) 
        return dim
    def getFeatureName(self,id):
        if hasProperty(self.domain, 'DimNames'):
            dim     = searchsorted(self.maxFeatureIDperDimension,id)
            #Find the index of the feature in the corresponding dimension
            index_in_dim = id
            if dim != 0:
                index_in_dim = id - self.maxFeatureIDperDimension[dim-1]
            print self.domain.DimNames[dim]
            f_name = self.domain.DimNames[dim] + '=' + str(index_in_dim)
    def featureType(self):
        return bool
