from Tools import *
from Domains import *
from Representations import *


F       = sp.lil_matrix([[1,1],[2,2],[3,3]])
p, n    = F.shape
a       = 4
A       = array([1,0,3])
M       = sp.lil_matrix((p,n*a))
print F.todense()
print A
for i in arange(a):
    rows = where(A==i)[0]
    print i,rows
    if len(rows):
        print F[rows,:].todense()
        print 'cols',i*n,(i+1)*n
        M[rows,i*n:(i+1)*n] = F[rows,:]
print M.todense()
