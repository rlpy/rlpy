from Tools import *
from Domains import *
from Representations import *

class iFDD_feature(object):
    index   = None        #Unique index Corresponding to its location in a vector (0 based!)
    f_set   = None        #A set of basic features that this feature corresponds to
    p1      = None        #Parent 1 feature index *Basic features have -1 for both nodes
    p2      = None        #Parent 2 feature index *Basic features have -1 for both nodes
    def __init__(self,potential):
            # This is the case where a single integer is passed to generate an initial feature
            index           = potential
            self.index      = index
            self.f_set      = frozenset([index])
            self.p1         = -1
            self.p2         = -1
    def show(self):
        printClass(self)
class X:
    dict = {}
    def __deepcopy__(self,memo):
        x = X()
        x.dict = deepcopy(self.dict)
        return x

f1      = iFDD_feature(1)
f2      = iFDD_feature(2)
x1      = X()
x1.dict[1] = f1
x1.dict[2] = f2

x2      = deepcopy(x1)
x1.dict[1].f_set = frozenset([10])
x1.dict[2].index = 20
print x1.dict[1].show()
print x1.dict[2].show()
print x2.dict[1].show()
print x2.dict[2].show()
  

