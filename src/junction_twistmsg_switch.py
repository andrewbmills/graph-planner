#!/usr/bin/env python
import sys
import numpy as np
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *

class VelocitySwitcher:
	def getBool(self, msg):
		self.switch = msg.data
		return

	def getVel1(self, msg):
		self.cmd_vel_1 = msg
		return

	def getVel2(self, msg):
		self.cmd_vel_2 = msg
		return

	def start(self):
		rate = rospy.Rate(self.rate) # 5Hz
		while not rospy.is_shutdown():
			rate.sleep()
			if self.switch:
				self.pub.publish(self.cmd_vel_2)
			else:
				self.pub.publish(self.cmd_vel_1)
				
		return

	def __init__(self):
		node_name = "cmd_vel_switch"
		rospy.init_node(node_name)
		self.rate = float(rospy.get_param("/cmd_vel_switch/rate", 1.0))

		# Subscribers
		rospy.Subscriber("cmd_vel_1", Twist, self.getVel1)
		rospy.Subscriber("cmd_vel_2", Twist, self.getVel2)
		rospy.Subscriber("switch", Bool, self.getBool)

		# Publishers
		self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

		# Initialize Subscription storage objects
		self.switch = False
		self.cmd_vel_1 = Twist()
		self.cmd_vel_2 = Twist()

if __name__ == '__main__':
	node = VelocitySwitcher()

	try:
		node.start()
	except rospy.ROSInterruptException:
		pass