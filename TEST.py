from Tools import *
from Representations import *
from Domains import *

a = array([[0,0],[2,4]])
b  = array([5,6])
aa = a + .1*eye(2)
print linalg.solve(aa,b)
print slinalg.lsmr(a,b)
