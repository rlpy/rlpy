from Tools import *
from Domains import *
from Representations import *


A = array([
           [1,2],
           [3,4],
           [1,4],
           [5,6],
           ])
pits = array([[1,4],[3,7]])
print [x in pits for x in A]