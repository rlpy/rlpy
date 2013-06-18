#!/usr/bin/env python

# Framework acting as a client.
import roslib; roslib.load_manifest('car_sim')
import numpy
import sys
import subprocess
import time
from Tools import *

import rospy
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import *
from car_sim.srv import *
from car_sim.msg import *

# This class assumes that you have both ROS properly installed AND Mark Cutter's car simulator.
# If the internal flag is True, the script will call the proper command to initialize a car simulation in Mark's simulator in a seperate process.
#    Note that this proccess does not terminate correctly and will still be running even if the Framework is killed. I do not adive using it
# If you want to initialize this simulation seperately use the command "roslaunch car_control control.launch veh:=RC num:=01 sim:=true"
#
# The visualize flag determines whether Mark's simulation should visualize the simulation. Use this command to initialize the visualizer: "roslaunch raven_rviz raven_world.launch"
#
# See the following two tutorials to get ROS communicating between multiple computers:
# http://www.ros.org/wiki/ROS/NetworkSetup
# http://www.ros.org/wiki/ROS/Tutorials/MultipleMachines
#
# Email me at gadgy@mit.edu if you have any questions, Elliott

class RC_Com():
    def __init__(self, internal=False):
        self.state = CarState(pose=Pose(position=Point(0.00001,0.00001,0.00001)))
	self.XMAX = 3/2.0
	self.XMIN = -3/2.0
	self.YMAX = 2/2.0
	self.YMIN =  -2/2.0
	if internal:
		subprocess.Popen("roslaunch car_control control.launch veh:=RC num:=01 sim:=true", shell=True)

    def Step(self, s,a,visualize=True):
        rospy.wait_for_service('RC01/run_step')
        try:
            step = rospy.ServiceProxy('RC01/run_step', RunStep)
	    omegaDes, turn = self.decodeAction(a)
	    #print omegaDes
	    #omegaDes, turn = 0.00001, 0.000001
	    #omegaDes, turn =1,-1
	    resp1 = step(self.state,0.2,2*omegaDes/0.035,turn,visualize) # 0.005
            ns = resp1.finalState
	    if ns.pose.position.x >= self.XMAX:
		ns.pose.position.x = self.XMAX
		ns.Vx = 0
	    elif ns.pose.position.x <= self.XMIN:
		ns.pose.position.x = self.XMIN
		ns.Vx = 0
	    if ns.pose.position.y >= self.YMAX:
		ns.pose.position.y = self.YMAX
		ns.Vy = 0
	    elif ns.pose.position.y <= self.YMIN:
		ns.pose.position.y = self.YMIN
		ns.Vy = 0
      	    self.state = ns
            speed = (ns.Vx**2 + ns.Vy**2)**0.5
	    quat_head = [ns.pose.orientation.x,ns.pose.orientation.y,ns.pose.orientation.z,ns.pose.orientation.w]
	    heading = euler_from_quaternion(quat_head)[2]
	    #print ns.pose.position.x, ns.pose.position.z
            new_state = numpy.array([ns.pose.position.x,ns.pose.position.y,speed,heading])
            #print new_state
            return new_state
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def resetState(self):
	self.state = CarState(pose=Pose(position=Point(0.00001,0.00001,0.00001)))

    def decodeAction(self, a):
        omegaDes,turn                = id2vec(a,[3,3])
        omegaDes                     -= 1
        turn                    -= 1
        return omegaDes+0.00000001, turn+0.00000001
