from Tools import *
from main import *
from Representations import *
from Domains import *
import timeit
from scipy.sparse.linalg.dsolve import spsolve, use_solver

def f1():
    x = random.random_integers(0,L-1,L)
    y = [L]*L
    return vec2id(x,y)
def f2():
    x = random.random_integers(0,L-1,L)
    y = [L]*L
    return vec2id2(x,y)
def foo(L,solver):
    o = ones(L)
    rows = random.random_integers(0,L-1,L)
    cols = random.random_integers(0,L-1,L)
    M = sp.csc_matrix((o,(rows,cols)),shape=(L,L))
    b = arange(L)
    x = solver(M,b)
random.seed(999)
L = 100000
t = timeit.Timer(stmt="foo(L,slinalg.lsmr)",setup="from __main__ import *")
print t.timeit(number=1) 
random.seed(999)
t = timeit.Timer(stmt="foo(L,slinalg.spsolve)",setup="from __main__ import *")
print t.timeit(number=1)  

