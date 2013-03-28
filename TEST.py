from Tools import *
from Domains import *
from Representations import *

class Y:
    def __init__(self,x):
        self.id = x
class X:
    dict = {}
    def __deepcopy__(self,memo):
        x = X()
        x.dict = deepcopy(self.dict)
        return x
y       = Y(array([1,2,3]))
x1      = X()
x1.dict[1] = y

x2      = deepcopy(x1)
x1.dict[1].id[0] = 10
print x2.dict[1].id  