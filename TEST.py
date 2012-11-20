from Tools import *
from Representations import *
from Domains import *

#A = array([[1,0],[0,1]])
#b = [2,1]
#print solveLinear(A,b)

A = ones((2,2))
b = A[:,1].copy()
b[0] = 0
print A
c = array([0,0])
A[1,:] = c
print A
c[0] = 1
print A