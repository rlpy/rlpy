from sklearn import svm
from Tools import *
from Domains import PitMaze

row  = array([0,0,1,3,1,0,0])
col  = array([0,2,1,3,1,0,0])
data = array([1,1,1,1,1,1,1])
A = sp.coo_matrix((data, (row,col)), shape=(4,4))
