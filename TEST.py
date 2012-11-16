from Tools import *
from Representations import *
from Domains import *
import timeit
from sets import ImmutableSet
from test.test_set import powerset

d = {}
x = [1,2,3]
for i in range(4):
    d[ImmutableSet([i])] = i

for c in powerset(x):
    candid = ImmutableSet(c)
    if d.get(candid) != None:
        print d[candid] 
    