from Tools import *
from main import *
from Representations import *
from Domains import *
import timeit

def foo():
    L = 100
    o = ones(L)
    rows = random.random_integers(0,L-1,L)
    cols = random.random_integers(0,L-1,L)
    M = sp.csc_matrix((o,(rows,cols)),shape=(L,L))
    b = arange(L)
    x = slinalg.lsmr(M,b)

random.seed(999)
t = timeit.Timer(stmt="foo()",setup="from __main__ import *")
print t.timeit(number=100)  


