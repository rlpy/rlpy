from Tools import *
from Domains import *
from Representations import *

A =  array([
                 [0,2],
                 [0,3],
                 [0,2]])

X = array([0,1])
Y = array([0,1])
V = array([100,100])
A[X,Y] = 1
print A