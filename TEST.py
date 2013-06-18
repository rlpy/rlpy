from Tools import *
from Representations import *
from Domains import *
import timeit

x = array([1,2,3,4,5,6])
x = x.reshape(2,-1)
x = hstack((x,zeros((2,1))))
x = x.reshape(1,-1).flatten()
print x
