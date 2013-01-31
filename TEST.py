from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

A = array([
           [1, 2, 3, 0],
           [1, 2, 3, 0]])  # All x's, then all y's
norms = apply_along_axis(linalg.norm, 0, A)
print A/norms.reshape(1,-1)