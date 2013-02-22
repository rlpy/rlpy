from Tools import *
from Domains import *
from Representations import *

V = array([0,0,0])
A = array([[1,2,3]])
B = array([[1,2,3],[4,5,6]])
S = sp.coo_matrix(A)
print S.shape
print A.shape
print B.shape
print matrix_mult(A,A.T)