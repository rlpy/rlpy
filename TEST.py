from Tools import *
from Representations import *
from Domains import *

L = 100
o = ones(L)
M = sp.csc_matrix((o,(range(L),range(L))),shape=(L,L))
b = arange(L)
x = slinalg.spsolve(M,b)
print x