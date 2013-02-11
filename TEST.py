from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

A = array([[1,2,3],
           [4,5,6]])
I = array([[0,1,0],
           [1,0,1]])
I = sp.coo_matrix(I)
print I.todense()!=1
print [I!=1]

#M = array([[[1,2],[3,4]], [[5,6],[7,8]], [[0,0],[0,0]]])
#print M
#A = [1,2]
#p = xrange(2)
#ind = zip(A,p)
#print ind
#X = M[1,0].reshape(-1)
#print X