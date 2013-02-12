from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

M = array([[1,2,3,4],
           [5,6,7,8]])
n = 2
A = array([1,0],dtype=integer)
A = kron(A,ones((1,n,),dtype=integer))[0]
print A 
print M[A,arange(len(A)),:].reshape(-1)

