from Representations import iFDD
from Representations import IndependentDiscretizationCompactBinary
import Domains
from Tools import Logger
import numpy as np

STDOUT_FILE         = 'out.txt'
JOB_ID              = 1
RANDOM_TEST         = 0
OUT_PATH            = 'Results/Temp'
logger              = Logger()
sparsify            = True
discovery_threshold = 1

def deterministic_test():
    discovery_threshold = 1
    sparsify = True
    domain = Domains.SystemAdministrator()
    initialRep  = IndependentDiscretizationCompactBinary(domain,logger)
    rep = iFDD(domain,logger,discovery_threshold,initialRep,
               debug=0, useCache=1, sparsify = sparsify)
    rep.theta = np.arange(rep.features_num*domain.actions_num)*10
    print 'Initial [0,1] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,1]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([0,1]))
    #rep.show()

    print rep.inspectPair(0,1, discovery_threshold+1)
    #rep.show()
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,1]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([21]))

    print 'Initial [2,3] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([2,3]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([2,3]))
    #rep.showCache()
    #rep.showFeatures()
    #rep.showCache()
    print 'Initial [0,20] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,20]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([0,20]))

    print 'Initial [0,1,20] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,1,20]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([20,21]))
    rep.showCache()
    # Change the weight for new feature 40
    rep.theta[40] = -100
    print 'Initial [0,20] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,20]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([0,20]))

    print 'discover 0,1,20'
    rep.inspectPair(20,rep.features_num-1, discovery_threshold+1)
    #rep.showFeatures()
    #rep.showCache()
    print 'Initial [0,1,20] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,1,20]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([22]))

    rep.showCache()
    print 'Initial [0,1,2,3,4,5,6,7,8,20] => ',
    ANSWER = np.sort(rep.findFinalActiveFeatures([0,1,2,3,4,5,6,7,8,20]))
    print ANSWER
    assert np.array_equal(ANSWER, np.array([2,3,4,5,6,7,8,22]))
    #rep.showCache()

import nose.tools
@nose.tools.nottest
def random_test():
    discovery_threshold
    TRIALS      = 200
    K           = 5 # number of randomly activated features
    domain      = Domains.PST()
    np.random.seed(999999999)
    initialRep  = IndependentDiscretizationCompactBinary(domain,logger)
    rep         = iFDD(domain,logger,discovery_threshold,initialRep,debug=0,useCache=1,iFDDPlus=0)
    rep.theta   = np.arange(rep.features_num*domain.actions_num)*10
    n           = rep.features_num
    for i in xrange(TRIALS):
        phi         = np.zeros(n,'bool')
        for j in xrange(K):
            ind = np.random.randint(0,n-1)
            phi[ind] = 1
            threshold = np.random.rand()*2-1
        print '%d: %0.2f >> %s' % (i+1, threshold, str(phi.nonzero()[0]))
        rep.discover(phi,threshold)
    rep.show()


