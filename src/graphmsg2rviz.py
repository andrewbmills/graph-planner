#!/usr/bin/env python
import sys
import numpy as np
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from gbm_pkg.msg import *

def euler2quaternion(euler):
	# Pre-allocate quaternion array
	q = Quaternion()
	# We assume "ZYX" rotation order
	c = np.cos(euler/2);
	s = np.sin(euler/2);
	# Calculate q
	q.w = c[0]*c[1]*c[2]+s[0]*s[1]*s[2]
	q.x = c[0]*c[1]*s[2]-s[0]*s[1]*c[2]
	q.y = c[0]*s[1]*c[2]+s[0]*c[1]*s[2]
	q.z = s[0]*c[1]*c[2]-c[0]*s[1]*s[2]
	return q

class GraphMsgPlotter:
	def getGraph(self, data): # Graph subscriber callback function
		# Erase the previous PoseArray messages
		self.explored_PoseArray = PoseArray()
		self.explored_PoseArray.header.frame_id = self.fixed_frame
		self.unexplored_PoseArray = PoseArray()
		self.unexplored_PoseArray.header.frame_id = self.fixed_frame

		# Udpate msgs with graph info
		for node in data.node:
			idx = int(node.id)
			if (int(node.id) == data.currentNodeId):
				newPose = Pose()
				newPose.position = node.position
				newPose.orientation = euler2quaternion(np.array([data.currentEdge, 0.0, 0.0]))
				self.current_edge_PoseStamped.pose = newPose

			for edge_num in range(node.nExploredEdge):
				# Add the edge to the corresponding PoseArray msg
				newPose = Pose()
				newPose.position = node.position
				newPose.orientation = euler2quaternion(np.array([node.exploredEdge[edge_num], 0.0, 0.0]))
				self.explored_PoseArray.poses.append(newPose)
			if node.nUnexploredEdge:
				for edge_angle in node.unexploredEdge:
					newPose = Pose()
					newPose.position = node.position
					newPose.orientation = euler2quaternion(np.array([edge_angle, 0.0, 0.0]))
					self.unexplored_PoseArray.poses.append(newPose)

		# Stamp the msgs
		self.explored_PoseArray.header.stamp = rospy.Time.now()
		self.unexplored_PoseArray.header.stamp = rospy.Time.now()
		self.current_edge_PoseStamped.header.stamp = rospy.Time.now()
		return

	def start(self):
		rate = rospy.Rate(self.rate) # 50Hz
		while not rospy.is_shutdown():
			rate.sleep()

			self.pub1.publish(self.explored_PoseArray)
			self.pub2.publish(self.unexplored_PoseArray)
			self.pub3.publish(self.current_edge_PoseStamped)
		return

	def __init__(self):
		node_name = "graphmsg2rviz"
		rospy.init_node(node_name)
		self.rate = float(rospy.get_param("/graphmsg2rviz/rate", 5.0))
		self.fixed_frame = str(rospy.get_param("/graphmsg2rviz/fixed_frame", "world"))

		# Subscribers
		rospy.Subscriber("graph", Graph, self.getGraph)

		# Publishers
		self.pub1 = rospy.Publisher("ExploredEdges", PoseArray, queue_size=10)
		self.pub2 = rospy.Publisher("UnexploredEdges", PoseArray, queue_size=10)
		self.pub3 = rospy.Publisher("CurrentEdge", PoseStamped, queue_size=10)

		# Published msg storage variables
		self.explored_PoseArray = PoseArray()
		self.explored_PoseArray.header.frame_id = self.fixed_frame
		self.unexplored_PoseArray = PoseArray()
		self.unexplored_PoseArray.header.frame_id = self.fixed_frame
		self.current_edge_PoseStamped = PoseStamped()
		self.current_edge_PoseStamped.header.frame_id = self.fixed_frame

if __name__ == '__main__':
	node = GraphMsgPlotter()

	try:
		node.start()
	except rospy.ROSInterruptException:
		pass