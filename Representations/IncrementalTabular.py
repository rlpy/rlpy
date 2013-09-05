#See http://acl.mit.edu/RLPy for documentation and future code updates

#Copyright (c) 2013, Alborz Geramifard, Robert H. Klein, and Jonathan P. How
#All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#Neither the name of ACL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################
# Developed by Alborz Geramiard Oct 30th 2012 at MIT #
######################################################
from Representation import Representation
import numpy as np
from copy import deepcopy


class IncrementalTabular(Representation):
    hash = None

    def __init__(self, domain, logger, discretization=20):
        self.hash           = {}
        self.features_num   = 0
        self.isDynamic      = True
        super(IncrementalTabular, self).__init__(domain, logger, discretization)

    def phi_nonTerminal(self, s):
        hash_id = self.hashState(s)
        id  = self.hash.get(hash_id)
        F_s = np.zeros(self.features_num, bool)
        if id is not None:
            F_s[id] = 1
        return F_s

    def pre_discover(self, s, terminal, a, sn, terminaln):
        return self._add_state(s) + self._add_state(sn)

    def _add_state(self, s):
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
            return 1
        return 0

    def __deepcopy__(self, memo):
        new_copy = IncrementalTabular(self.domain, self.logger, self.discretization)
        new_copy.hash = deepcopy(self.hash)
        return new_copy

    def featureType(self):
        return bool
