from Tools import *
from Representations import *
from Domains import *
from scipy.stats import *

A = ndarray([1,1,3])
A[0,0,:] = array([1,1,1])
print sem(A,axis=2)