from Tools import *
from Representations import *
from Domains import *
import timeit

def foo():
    n       = 3
    a       = empty((n**2,2))
    index   = 0 
    for i in range(n):
        for j in range(n):
            a[index,:] = [i,j]
            index += 1
    return a
def goo():
    n = 3
    return array([[a,b] for a in range(n) for b in range(n)])

def hoo():
    n = 3
    return [[a,b] for a in range(n) for b in range(n)]

t = timeit.Timer(stmt="foo()",setup="from __main__ import *")
print t.timeit(number=10000)  
t = timeit.Timer(stmt="goo()",setup="from __main__ import *")
print t.timeit(number=10000)
t = timeit.Timer(stmt="hoo()",setup="from __main__ import *")
print t.timeit(number=10000)  