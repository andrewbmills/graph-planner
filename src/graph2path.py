#!/usr/bin/env python
import sys
import numpy as np
import rospy
import copy
from std_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from gbm_pkg.msg import *
from dijkstra import GraphSolver

def angleDiff(a, b):
	# Computes a-b, preserving the correct sign (counter-clockwise positive angles)
	# All angles are in degrees
	a = (360000 + a) % 360
	b = (360000 + b) % 360
	d = a - b
	d = (d + 180) % 360 - 180
	return d

def wrapToPi(a):
	# Wraps an angle to sit on the interval (-pi,pi)
	a = (a + np.pi) % (2*np.pi) - np.pi
	return a

def euler2quaternion(euler):
	# Pre-allocate quaternion array
	q = [0, 0, 0, 0]
	# We assume "ZYX" rotation order
	c = np.cos(euler/2);
	s = np.sin(euler/2);
	# Calculate q
	q[0] = c[0]*c[1]*c[2]+s[0]*s[1]*s[2]
	q[1] = c[0]*c[1]*s[2]-s[0]*s[1]*c[2]
	q[2] = c[0]*s[1]*c[2]+s[0]*c[1]*s[2]
	q[3] = s[0]*c[1]*c[2]-c[0]*s[1]*s[2]
	return q

def check_symmetric(a, rtol=1e-05, atol=1e-08):
	# Checks if a matrix is close to symmetric given the relative and absolute tolerances
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class graph2path:
	def maintainCrumbs(self, new_id):
		# This function finds the last time the current id occured in the breadcrumb list
		# Then it shortens the breadcrumb list up to that id so that the robot doesn't waste time going home
		idx = len(self.breadcrumbs)-1
		while (not self.breadcrumbs[idx] == new_id):
			if (idx == 0):
				self.breadcrumbs.append(new_id)
				return
			idx = idx - 1
		self.breadcrumbs = self.breadcrumbs[:idx+1]
		return

	def getOdom(self, data): # Odom subscriber callback function
		self.position = data.pose.pose.position
		q = Quaternion()
		q = data.pose.pose.orientation
		self.yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
		return

	def getGraph(self, data): # Graph subscriber callback function
		self.first_graph_msg = True
		self.graphmsg = data

	def readGraph(self, data):
		self.node_position_list = []
		# Generates an adjacency graph (numpy 2d array) from Graph.msg data
		n = data.size
		Adj = np.zeros((n,n,2)) # Adjacency matrix (First nxn is costs and second one is angles)
		self.unexplored_edges = [] # a (p x e+1) list of unexplored edges.  Column 1 is the node id and columns 2 to e+1 are the edge angles
		self.current_node = 0
		min_node_dist_squared = float("inf")
		for node in data.node:
			i = int(node.id)
			self.node_position_list.append(node.position)
			# Check if this node is the closest node to the robot's current odom
			dist_2 = (self.position.x - node.position.x)**2 + (self.position.y - node.position.y)**2
			if (dist_2 <= min_node_dist_squared):
				min_node_dist_squared = dist_2
				self.current_node = i
			for edge_num in range(len(node.neighborId)):
				j = node.neighborId[edge_num]
				cost = node.edgeCost[edge_num]
				angle = node.exploredEdge[edge_num]
				try:
					if (Adj[i,j,0] < 0.01) or (cost < Adj[i,j,0]):
						Adj[i,j,0] = cost
						Adj[i,j,1] = angle
				except:
					print(data)
					rospy.logwarn("Graph message ids out of bounds.")
			if node.nUnexploredEdge:
				self.unexplored_edges.append([i, node.unexploredEdge])

		# self.current_node = data.currentNodeId
		self.current_node_position.point = data.node[self.current_node].position
		self.current_node_position.header.stamp = data.header.stamp

		self.A = np.copy(Adj)
		if not check_symmetric(self.A[:,:,0]):
			print("A is not symmetric, printing graph msg:")
			print(data)

		# Add the current node to the history of nodes if it's different than the last added
		if (self.node_history):
			if (not self.node_history[-1] == self.current_node):
				self.node_history.append(self.current_node)
				# Maintain breadcrumb path back to home
				self.maintainCrumbs(self.current_node)
		else:
			self.node_history.append(self.current_node)
			self.breadcrumbs.append(self.current_node)
		return

	def getTask(self, data):
		self.task = str(data.data)
		if (self.task == "Report"):
			self.task = "Home"
		return

	def findPath(self):
		# Copy adjacency matrix
		Adj = np.copy(self.A)
		n = len(Adj)
		print("current node %d" % (self.current_node))
		node_list = []
		turn_list = []
		if (n>1):
			print(Adj[:,:,0])
			g = GraphSolver()
			try:
				parent, dist = g.dijkstra(Adj[:,:,0], self.current_node)
			except:
				rospy.logwarn("Graph msg not fully connected.")
				return [], []
		else:
			print("Less than two nodes in the graph.")
			return [], []
		# Find the goal id
		if (self.task == "Home"):
			# Home node id
			goal_id = 0
		else:
			# Closest node with an unexplored edge
			# print(self.unexplored_edges)
			goal_list = np.array([row[0] for row in self.unexplored_edges], dtype=np.int16)
			if (len(goal_list)>0):
				# print(goal_list)
				dist = np.array(dist)
				# print(dist)
				destination_edge_idx = np.argmin(dist[goal_list])
				goal_id = goal_list[destination_edge_idx]
				# print(dist[goal_list])
				# print(goal_id)
			else:
				print("No unexplored_edges, vehicle is heading home.")
				goal_id = 0

		# print(parent)
		# print(dist)
		# print(goal_id)
		# Extract path to goal_id from parent list
		j = goal_id
		# print(type(j))
		node_list.append(j)
		while (not parent[j] == -1):
			node_list.insert(0, parent[j])
			j = parent[j]

		# print(self.unexplored_edges)


		if (len(node_list) > 1):
			turn_list = self.turns(node_list, Adj)
			arriving_angle = wrapToPi(Adj[node_list[-1], node_list[-2], 1] + np.pi)
			if (self.task == "Home") or (goal_id == 0):
				# Just go straight when you get home
				turn_list.append(arriving_angle)
			else:
				# Add the turn at the goal node as the minimum turning angle from the arriving yaw
				delta_min = 361.0
				if self.turn_biasing:
					bias_angle = np.arctan2(self.node_position_list[node_list[-1]].y - self.position.y, self.node_position_list[node_list[-1]].x - self.position.x)
				for goal_angle in self.unexplored_edges[destination_edge_idx][1]:
					if self.turn_biasing:
						delta = np.abs(angleDiff((180.0/np.pi)*goal_angle, (180.0/np.pi)*bias_angle))
					else:
						delta = np.abs(angleDiff((180.0/np.pi)*goal_angle, (180.0/np.pi)*arriving_angle))
					if (delta < delta_min):
						delta_min = delta
						goal_angle_min = goal_angle
				turn_list.append(goal_angle_min)
		else:
			# If the vehicle is at that node, then proceed.  If not, the turn list should be empty.
			if (self.at_a_node.data):
				if (self.task == "Home") or (goal_id == 0):
					# Just go straight since you're at home
					turn_list.append(self.yaw)
				else:
					# Add the turn at the goal node as the minimum turning angle from the current yaw
					delta_min = 361.0
					if self.turn_biasing:
						bias_angle = np.arctan2(self.turn_goal_y - self.position.y, self.turn_goal_x - self.position.x)
					for goal_angle in self.unexplored_edges[destination_edge_idx][1]:
						if self.turn_biasing:
							delta = np.abs(angleDiff((180.0/np.pi)*goal_angle, (180.0/np.pi)*bias_angle))
						else:
							delta = np.abs(angleDiff((180.0/np.pi)*goal_angle, (180.0/np.pi)*self.yaw)) 
						if (delta < delta_min):
							delta_min = delta
							goal_angle_min = goal_angle
					turn_list.append(goal_angle_min)

		print(turn_list)

		return node_list, turn_list

	def turns(self, node_list, Adj):
		turn_list = []
		for i in range(len(node_list)-1):
			turn = Adj[node_list[i], node_list[i+1], 1]
			if turn == 0:
				rospy.logwarn("Planned path contained unconnected nodes")
				turn_list = []
				break
			else:
				turn_list.append(turn)
		return turn_list

	def setTurnCommand(self, goal_angle):
		delta_angle = angleDiff((180.0/np.pi)*goal_angle, (180.0/np.pi)*self.yaw)
		print("delta_angle = %0.2f degrees" % (delta_angle))
		if np.abs(delta_angle) > 10.0:
			self.turn_command.linear.x = 0.0
		else:
			self.turn_command.linear.x = self.speed
		self.turn_command.angular.z = np.sign(delta_angle)*max(abs(self.turn_gain*((np.pi/180.0)*delta_angle)), 0.2)
		print("Turn command = %0.2f rad/s" % self.turn_command.angular.z)
		return

	def pathHome(self):
		# Copy adjacency matrix
		Adj = np.copy(self.A)
		node_list = self.breadcrumb.reverse()
		turn_list = self.turns(node_list, Adj)
		return node_list, turn_list

	def setPoseMsg(self, euler):
		q = euler2quaternion(euler)
		self.next_turn_pose.pose.position = self.position
		self.next_turn_pose.pose.orientation.w = q[0]
		self.next_turn_pose.pose.orientation.x = q[1]
		self.next_turn_pose.pose.orientation.y = q[2]
		self.next_turn_pose.pose.orientation.z = q[3]
		self.next_turn_pose.header.stamp = rospy.Time.now()
		self.next_turn_pose.header.frame_id = self.fixed_frame
		return

	def setPoseArrayMsg(self, nodes, turns):
		self.turn_list_poses = PoseArray()
		self.turn_list_poses.header.stamp = rospy.Time.now()
		self.turn_list_poses.header.frame_id = self.fixed_frame
		if (len(nodes) < 1) or (len(turns) < 1):
			return
		for i in range(len(nodes)):
			try:
				newPose = Pose()
				newPose.position = self.node_position_list[nodes[i]]
				q = euler2quaternion(np.array([turns[i], 0.0, 0.0]))
				newPose.orientation.w = q[0]
				newPose.orientation.x = q[1]
				newPose.orientation.y = q[2]
				newPose.orientation.z = q[3]
				self.turn_list_poses.poses.append(newPose)
			except:
				continue
		return

	def start(self):
		rate = rospy.Rate(self.rate)
		while not rospy.is_shutdown():
			rate.sleep()
			turn_list = []
			if (self.first_graph_msg):
				# Convert graph msg from callback into readable data
				try:
					self.readGraph(copy.deepcopy(self.graphmsg))
				except:
					rospy.logwarn("Graph message could not be read")
					continue
				if (self.task == "Home_slow"):
					node_list, turn_list = self.pathHome()
					print("Robot is heading home now.")
				else:
					node_list, turn_list = self.findPath()
				# Check if robot is at a node
				dist = np.sqrt((self.position.x - self.current_node_position.point.x)**2 + (self.position.y - self.current_node_position.point.y)**2)
				if (dist <= self.at_a_node_radius):
					self.at_a_node.data = True
				else:
					self.at_a_node.data = False
			else:
				rospy.loginfo("Waiting for first graph message")
				continue

			if (len(turn_list)>0):
				self.next_turn.data = turn_list[0]
				self.setPoseMsg(np.array([turn_list[0], 0.0, 0.0]))
				if (self.at_a_node.data):
					print("Robot is at a node.  Turn to %0.2f deg.  Current heading is %0.2f deg" % ((180/np.pi)*turn_list[0], (180/np.pi)*self.yaw))
					self.setTurnCommand(turn_list[0])
				else:
					if (len(turn_list) > 1):
						print("Next turn is %0.2f deg at node %d." % ((180/np.pi)*turn_list[1], node_list[1]))
					else:
						print("The robot has left from a node with an unexplored edge.  Will update command at next junction.")
			else:
				self.setPoseMsg(np.array([self.yaw, 0.0, 0.0]))
				print("The robot has left from a node with an unexplored edge.  Will update command at next junction.")
				self.next_turn.data = -10.0
				self.setTurnCommand(self.yaw)

			self.pub2.publish(self.next_turn)
			self.pub4.publish(self.next_turn_pose)
			self.setPoseArrayMsg(node_list, turn_list)
			self.pub5.publish(self.turn_list_poses)
			self.pub1.publish(self.at_a_node)
			self.pub3.publish(self.turn_command)
			self.pub6.publish(self.current_node_position)
		return

	def __init__(self):
		node_name = "graph2path"
		rospy.init_node(node_name)
		self.rate = float(rospy.get_param("graph2path/rate", 3.0))
		self.fixed_frame = str(rospy.get_param("graph2path/fixed_frame", "world"))
		self.speed = float(rospy.get_param("graph2path/speed", 1.0))
		self.at_a_node_radius = float(rospy.get_param("graph2path/node_radius", 3.0))
		self.turn_biasing = bool(rospy.get_param("graph2path/turn_biasing", False))
		self.turn_goal_x = float(rospy.get_param("graph2path/turn_goal_x", 5000.0))
		self.turn_goal_y = float(rospy.get_param("graph2path/turn_goal_y", 0.0))

		# Subscribers
		rospy.Subscriber("graph", Graph, self.getGraph)
		self.first_graph_msg = False
		rospy.Subscriber("odometry", Odometry, self.getOdom)
		rospy.Subscriber("task", String, self.getTask)

		# Publishers
		self.pub1 = rospy.Publisher("at_a_node", Bool, queue_size=10)
		self.pub2 = rospy.Publisher("next_turn", Float32, queue_size=10)
		self.pub3 = rospy.Publisher("cmd_turn_junction", Twist, queue_size=10)
		self.pub4 = rospy.Publisher("next_turn_pose", PoseStamped, queue_size=10)
		self.pub5 = rospy.Publisher("turn_list_poses", PoseArray, queue_size=10)
		self.pub6 = rospy.Publisher("junction_center", PointStamped, queue_size=10)

		# Initialize Subscription storage objects
		self.position = Point()
		self.current_node_position = PointStamped()
		self.current_node_position.header.frame_id = self.fixed_frame
		self.yaw = 0.0
		self.task = "Explore"
		self.graphmsg = Graph()
		# self.task = "Home"
		# self.task = "Home_slow"

		# Initialize Publisher message objects
		self.at_a_node = Bool()
		self.at_a_node.data = False
		self.next_turn = Float32()
		self.next_turn.data = -10.0
		self.turn_command = Twist()
		self.next_turn_pose = PoseStamped()
		self.turn_list_poses = PoseArray()
		self.turn_list_poses.header.frame_id = self.fixed_frame

		# Graph and node history holder arrays
		self.A = np.array([])
		self.node_history = []
		self.breadcrumbs = [] # a way home

		# Heading controller
		self.turn_gain = 0.1

if __name__ == '__main__':
	node = graph2path()

	try:
		node.start()
	except rospy.ROSInterruptException:
		pass
