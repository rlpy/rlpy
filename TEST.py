from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

A  = sp.csr_matrix((3,3),dtype='bool')
b  = array([0,1,0])
#c  = array([1,2,3])
#b_sp = sp.csr_matrix(b,dtype='bool')
#o  = b_sp.T*b_sp
#print o.todense(),type(o), size(o)
#A = A + o
#print A
#print type(A)
#inner = b_sp*c.T
#print inner,type(inner), size(inner)
#print inner.item()
#
#print sp_dot_array(b_sp,c)

print issubclass(b.dtype.type,integer) or issubclass(b.dtype.type,bool_)
print b.dtype.type
print A.dtype.type