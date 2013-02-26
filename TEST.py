from Tools import *
from Domains import *
from Representations import *


F = sp.csr_matrix([[1,1],[2,2],[3,3]])
p, n = F.shape
a = 4
A       = array([1,0,3])
rows    = arange(p) 
A_ind   = zeros((p,a))
A_ind[rows,A] = 1

output = zeros((p,n*a))
output(rows,)
print F*A_ind