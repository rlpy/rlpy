from Tools import *
from Representations import *
from Domains import *

F_s = array([1,0,0])
A = zeros(5)
A[3] = 1
F_sa = kron(A,F_s)
print F_sa