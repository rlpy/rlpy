# example.dpomdp
# -----------
# Frans Oliehoek - faolieho@science.uva.nl
# 2006-08-03
#
#
# This is a Dec-POMDP (.dpomdp) test file, which demonstrates most syntactic 
# constructs.
# Non-demonstrated constructs are mentioned in the comments, therefore this
# document also serves as documentation for the .dpomdp file format.
#
# First we note a few important things:
#
# 1) the file format is case-sensitive
# 2) the first entries in this file are:
#	agents
#	discount
#	values
#	states
#	start
#	actions
#	observations
#    These entries should *all* be present exactly *once* and in above *order*.
# 3) Other inconsistencies with cassandra's format are mentioned throughout
#    this file.
# 4) The file is 'newline sensitive': new lines should start at the correct
#    places! 
#    (i.e. no empty lines allowed! if you want space to this:
#
#    )
# 5) Comments should start on a new-line.
# 6) Identifiers consist of a letter followed by alphanumerics and '-' and '_'
# 7) as a general rule of thumb: 1 number (int or real) is placed on the same
#    line, multiple numbers (a vector) or matrices, start on a new line.
#    The keywords "uniform" and "identity" represent vectors/matrices and thus
#    also are placed on a new line.
# 8) Unlike cassandra's format, at the end of transitions, observations and 
#    reward specifications, right before the number we also require a colon.
#    I.e.:
#	R: open-right open-right : tiger-left : * : * : 20
#    This is because in 
#	R: open-right open-right : tiger-left : * : * 20
#    the 20 could also indicate the 20th individual observation of agent 2.
#
# Allright, here we go!
#
#The agents. 
#----------
#Either 1) the number of agents: 
#   agents: %d
#or 2) a list of agent identifiers, e.g.:
#   agents: agent1_name name-of-agent2 ... //TODO:this is not implem. yet!
#agents: agent1 the_second-agent 
agents: 2 
#
#the discount factor. 
#-------------------
#As this is dependent on the planning horizon (typically 
#1.0 for finite horizon problems) this is more a property of the planning 
#problem than one of the dec-pomdps. Nevertheless we include it to be as much
#compatiable with cassandra and umass as possible. 
#   discount: %f 
discount: 1.0
#
#values
#------
#reward or cost reward type...
#   values: [ reward, cost ] 
values: reward
#
#the states declaration
#-----------------------
#the number of states or a state list
#   states: [ %d, <list of states> ] 
states: state-one s2     
#
#The initial state distribution
#------------------------------
#NOTE: unlike cassandra, this is not optional and the position is BEFORE the 
#specification of the actions and observations.
#There are 4 ways to specify the starting distribution:
#   * enumerate the probabilities for each state,
#   * specify a single starting state,
#   * give a uniform distribution over states, or
#   * give a uniform distribution over a subset of states.
#
#In the last case, you can either specify a list of states too be included,
#or a list of states to be excluded. 
#Examples of this are:
#   start: 
#   0.3 0.1 0.0 0.2 0.5
#or   
#   start: 
#   uniform
#or
#   start: first-state
#or
#   start: 5
#or
#   start include: first-state third state
#or
#   start include: 1 3 indices-mixed-with-names s2 4
#or
#   start exclude: fifth-state seventh-state
start exclude: s2
#start: 
#0.3 0.1 
#
#The actions declarations
#------------------------
#the  (number/list of) actions for each of the agents on a separate line
#   actions: 
#   [ %d, <list of actions> ] 
#   [ %d, <list of actions> ] 
#   ...
#   [ %d, <list of actions> ] 
#
#   e.g. 3 named actions for agent 1 - 2 unnamed actions for agent 2:
actions: 
agent1-a1 a12 a13
2
#the (number/list of) observations for each of the agents on a separate line
#   observations: 
#   [ %d, <list of observations> ]
#   [ %d, <list of observations> ]
#   ...
#   [ %d, <list of observations> ]
observations: 
2
hear-left hear-right
#Joint actions
#-------------
#Although not explicitly specified, joint actions are an important part of the 
#model. Their syntax is explained here. There are (will be) 3 ways to denote a
#joint action:
#
# 1) by index 
# In accordence with C++ conventions, indices start at 0.
# Because, earlier in this file, only individual actions are specified, we use
# a convention as to how the joint actions (and joint observations) are 
# enumerated:
#    jaI    first agent <- individual indices -> last agent
#      0		0   0   .....	0	    0
#      1		0   0   .....	0	    1
#|O_n|-1		0   0	.....	0     |O_n|-1
#
# (note that this ordering is enforced by the way joint observation objects
# are created in the implementation. In particular see
#  MADPComponentDiscreteObservations::ConstructJointObservations )
#
# 2) by individual action components
#
# For now a joint action is specified by its individual action components, these
# can be indices (again, starting at 0), names or combinations thereof. E.g. 
# following the action specification above, a valid joint action would be 
# "a12 1" 
# It is also allowed to use the '*' wildcard. E.g., "a12 *" would denote all 
# joint actions in which the first agent performes action a12.
#
# 3) the wild-card
#It is also possible to use joint action "*" denoting all joint actions.
#
#
#Joint Observations
#------------------
#Joint observations use the same syntax as joint actions.
#
#Transition probabilities
#------------------------
#This explains the syntax of specifying transition probabilities. In the below
#<a1 a2...an> denotes a joint action, which can be specified using the syntax as
#explained above.
#
# NOTE: in contrast to Tony's pomdp file format, we also require a colon after
# the end-state.
#
#   T: <a1 a2...an> : <start-state> : <end-state> : %f
#or
#   T: <a1 a2...an> : <start-state> :
#   %f %f ... %f			    P(s_1'|ja,s) ... P(s_k'|ja,s)
#or
#   T: <a1 a2...an> :			    this is a |S| x |S| matrix
#   %f %f ... %f			    P(s_1'|ja,s_1) ... P(s_k'|ja,s_1)
#   %f %f ... %f			    ...
#   ...					    ...
#   %f %f ... %f			    P(s_1'|ja,s_k) ... P(s_k'|ja,s_k)
#or
#   T: <a1 a2...an> 
#   [ identity, uniform ]
#
#T: * : 
#uniform
T: * :
uniform
T: a12 1 : s2 : 
0.4 0.6
#T: 1 2 :
#identity 
T: 1 2 :
0.2 0.8
0.6 0.5
T: * * : * : * : 0.5
T: 1 1 : 0 : 1 : 0.33333
T: 1 1 : 0 : 0 : 0.66666
T: 1 : 0 : 
0.123 0.876
T: 0 0 : 0 : 
0.11111 0.88888
T: 2 :
identity 
T: 3 :
0.2222 0.2233
0.2244 0.2255
#Observation probabilities
#    O: <a1 a2...an> : <end-state> : <o1 o2 ... om> : %f
#or
#    O: <a1 a2...an> : <end-state> :
#    %f %f ... %f	    P(jo_1|ja,s') ... P(jo_x|ja,s')
#or
#    O:<a1 a2...an> :	    - a |S|x|JO| matrix
#    %f %f ... %f	    P(jo_1|ja,s_1') ... P(jo_x|ja,s_1') 
#    %f %f ... %f	    ... 
#    ...		    ...
#    %f %f ... %f	    P(jo_1|ja,s_k') ... P(jo_x|ja,s_k') 
#this uses the fact that earlier specified entries can be overwritten - this IS explicitly mentioned by cassandra (but not by umass) and is very convenient...
#TODO: the current format does not allow for independent transitions/observations - this might also be convenient...
O: * * :
uniform
O: a12 1 : state-one : * * : 0.002
O: a12 1 : state-one : 0 hear-left : 0.7225
O: 0 0 : * : * : 0.9
O: 0 1 : s2 : 0 hear-right : 0.1275
O: 0 1 : 0 : 1 hear-left : 0.1275
O: 1 0 : 1 : * hear-right : 0.0225
O: 1 1 : 1 : 1 * : 0.7225
O: 2 0 : 0 : 0 hear-right  : 0.1275
O: 2 1 : 0 : 0 0 : 0.1275
O: 2 1 : 1 : * : 0.0225
O: * 1 :
uniform
O: a12 1 : s2 : 
0.1 0.1 0.1 0.6 
O: 1 2 :
0.1 0.8 0.05 0.05
0.6 0.3 0.06 0.04
#The rewards
#    R: <a1 a2...an> : <start-state> : <end-state> : <o1 o2 ... om> : %f
#or
#    R: <a1 a2...an> : <start-state> : <end-state> :
#    %f %f ... %f
#or
#    R: <a1 a2...an> : <start-state> :
#    %f %f ... %f 
#    %f %f ... %f 
#    ...
#    %f %f ... %f
#
#here the matrix has nrStates rows and nrJointObservations columns.
#
#Typical problems only use R(s,ja) which is specified by:
#   R: <a1 a2...an> : <start-state> : * : * : %f
R: * : 1 : 3 :
4.3 34.2 253 12
R: * : s2 : 
4.3 34.2 253 12
5.3 2352 2.3 1.2
R: a12 0 : * : * : * : -2
R: 0 0 : 0 : * : * : -50
R: 0 1: 1 : * : * : +20
R: 0 2: 1 : * : * : 20
R: 1 0: 0 : * : * : -100
R: 1 1: 1 : * : * : -10
R: 1 2: 0 : * : * : -101
R: 2 0: 1 : * : * : -101
R: 2 1 : 1 : * : * : 9