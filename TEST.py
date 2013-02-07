from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

l = Logger()
d = BlocksWorld(logger = l)
r = IndependentDiscretization(domain = d,logger = l)
print r.getFeatureName(1)