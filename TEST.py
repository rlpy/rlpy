from Tools import *
from Domains import *
from Representations import *

A = sp.lil_matrix([[1,0],[0,1]])
print A.shape
print A.reshape((1,4))