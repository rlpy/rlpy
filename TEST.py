from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

A = array([[1,2,3],
           [4,5,6]])
I = array([[0,1,0],
           [1,0,1]])
I = sp.coo_matrix(I)
r = I.row
c = I.col
print A.T[c,r]