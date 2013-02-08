from sklearn import svm
from Tools import *
from Domains import *
from Representations import *

# make a mask for the invalid_actions
# build a matrix p-by-a where in each row the missing action is 1
# Example: 2 actions, 3 states
# possibleActions(s1) = 0
# possibleActions(s2) = 1
# possibleActions(s3) = 0,1
# output =>  0 1
#            1 0
#            0 0 
        
all_phi_s = array([
                   [1, 0],
                   [0, 2],
                   [3, 3]]
                   )
theta = array([1,2,3,4])

p,n = 3,2
max_a = 2
action_mask = ones((p,max_a))
action_mask[0,1] = 0
action_mask[1,0] = 0
action_mask[2,0:2] = 0
print action_mask
all_phi_s_a = kron(eye(max_a,max_a),all_phi_s) #all_phi_s_a will be ap-by-an
print all_phi_s_a
print all_phi_s_a.shape, theta.shape
all_q_s_a   = dot(all_phi_s_a,theta.T)           #ap-by-1
print all_q_s_a
all_q_s_a   = all_q_s_a.reshape((-1,max_a))    #a-by-p
print all_q_s_a
all_q_s_a   = ma.masked_array(all_q_s_a, mask=action_mask)
best_action = argmax(all_q_s_a,axis=1)
print best_action
action_slice = zeros((max_a,p))
action_slice[best_action,xrange(p)] = 1
# Build the slices in each row
# best_action = [1 0 1] with 2 actions and 3 samples
# build: 
# 0 1 0
# 1 0 1
print action_slice
#now expand each 1 into size of the features (i.e. n)
action_slice = kron(action_slice,ones((1,n*max_a)))
print "action_slice"
print action_slice
print "phi_s_a"
print all_phi_s_a
all_phi_s_a = all_phi_s_a.reshape((max_a,-1))
print "all_phi_s_a"
print all_phi_s_a
# with n = 2, and a = 2 we will have:
# 0 0 0 0 1 1 1 1 0 0 0 0
# 1 1 1 1 0 0 0 0 1 1 1 1
# now we can select the feature values
phi_s_a = all_phi_s_a.T[action_slice.T==1]
phi_s_a = phi_s_a.reshape((p,-1))
print 'final'
print phi_s_a
 
