from sklearn import svm
from Tools import *
from Domains import PitMaze

numDims = 2
numSamples = 10
y = zeros(numSamples)
X = random.randn(numSamples,numDims)
bebfApprox = svm.SVR(kernel='rbf', degree=3, C=1.0) # support vector regression
                                         # C = penalty parameter of error term, default 1
y = [0.0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3,0.4,0.2,0.1]
bebfApprox.fit(X,y)
feature = bebfApprox

print X
print y

print feature.predict([0,0])