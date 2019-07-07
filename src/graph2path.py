#!/usr/bin/env python
import sys
import numpy as np
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from gbm_pkg.msg import *
from dijkstra import GraphSolver

class graph2path:
	def getPosition(self, data): # Position subscriber callback function
		self.position = data.pose.pose.position
		return

	def getGraph(self, data): # Graph subscriber callback function
		# Generates an adjacency graph (numpy 2d array) from Graph.msg data
		n = data.size
		self.A = np.zeros((n,n,2)) # Adjacency matrix (First nxn is costs and second one is angles)
		self.unexplored_edges = [] # a (p x e+1) list of unexplored edges.  Column 1 is the node id and columns 2 to e+1 are the edge angles
		for node in data.node:
			i = node.id
			for edge_num in range(node.nExploredEdge):
				j = node.neighborId[edge_num]
				cost = node.edgeCost[edge_num]
				angle = node.exploredEdge[edge_num]
				if (self.A[i,j,0] == 0) or (cost < self.A[i,j,0]):
					self.A[i,j,0] = cost
					self.A[i,j,1] = angle
			if node.nUnexploredEdge:
				self.unexplored_edges.append([i, node.unexploredEdge])

		self.unexplored_edges = np.array(self.unexplored_edges)
		self.current_node = data.currentNodeId
		self.current_edge = data.currentEdge
		return

	def getTask(self, data):
		self.task = str(data.data)
		return

	def findPath(self):
		n = len(self.A)
		print(self.A[:,:,0])
		print("current node %d" % (self.current_node))
		if (n>1):
			g = GraphSolver()
			g.dijkstra(self.A[:,:,0], self.current_node)
		else:
			print("Less than two nodes in the graph.")

		return
		# goalList = []
		# if self.Task == "Home":
		# 	goalList.append(0)
		# if self.Task == "Explore":
		# 	for goal in self.unexplored_edges:
		# 		goal_node = goal[0]
		# 		goal_angle = goal[1]


	def start(self):
		rate = rospy.Rate(self.rate) # 50Hz
		while not rospy.is_shutdown():
			rate.sleep()
			self.findPath()
			self.pub1.publish(self.at_a_node)
			self.pub2.publish(self.next_turn)
		return

	def __init__(self):
		node_name = "graph2path"
		rospy.init_node(node_name)
		self.rate = 1.0

		# Subscribers
		rospy.Subscriber("/node_skeleton/graph", Graph, self.getGraph)
		rospy.Subscriber("/X1/odometry", Odometry, self.getPosition)
		rospy.Subscriber("task", String, self.getTask)

		# Publishers
		self.pub1 = rospy.Publisher("at_a_node", Bool, queue_size=10)
		self.pub2 = rospy.Publisher("next_turn", Float32, queue_size=10)

		# Initialize Subscription storage objects
		self.position = Point()
		self.task = "Explore"
		# self.task = "Home"

		# Initialize Publisher message objects
		self.at_a_node = Bool()
		self.at_a_node.data = False
		self.next_turn = Float32()
		self.next_turn.data = 0.0

		self.A = np.array([])

if __name__ == '__main__':
	node = graph2path()

	try:
		node.start()
	except rospy.ROSInterruptException:
		pass