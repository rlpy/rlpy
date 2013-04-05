from Tools import *
from Domains import *
from Representations import *
from fractions import Fraction

class C(object):
    hash = {}
    def __deepcopy__(self,memo):
        c = C()
        c.hash = deepcopy(self.hash)
        return c
c1 = C()
c1.hash[1] = 3
c2 = deepcopy(c1)
c1.hash[2] = 3
print c1.hash,c2.hash