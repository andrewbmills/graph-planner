#!/usr/bin/env python
import sys
import numpy as np
import rospy
import time
from itertools import chain
import cv2
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from nav_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from gbm_pkg.msg import *
import sensor_msgs.point_cloud2 as pc2
# My own wrapped thinning function
import sys
# sys.path.insert(0, '../include/')
# import thin_ext as cppthin
from skimage.morphology import skeletonize, thin
from skimage import measure
# image convolution library
from scipy import ndimage

def findNeighbors(i, j, skel):
	kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
	rows, cols = np.shape(skel)
	nh = skel[(i-1):(i+2),(j-1):(j+2)] # neighborhood
	nh_ids = np.transpose(np.nonzero(np.multiply(nh,kernel)))
	neighbors = np.transpose([(i-1) + nh_ids[:,0], (j-1) + nh_ids[:,1]])
	neighbors = neighbors[:,0]*cols + neighbors[:,1]
	return neighbors

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

class node_skeleton:
	def getPosition(self, data):
		self.position = data.pose.pose.position
		self.position.z = self.position.z  + 3.0*self.voxel_size
		self.position_get_first = True
		return

	def getOccupancyGrid(self, data): # nav_msgs/OccupancyGrid Subscriber
		self.occupancy_grid_msg = data
		self.map_get = True
		self.map_get_first = True
		return

	def filterNodes(self, nodes):
		# Filter out redundant nodes
		rows, cols = np.shape(nodes)
		added_indices = []
		nodes_filtered = []
		for i in range(rows):
			if i not in added_indices:
				d = np.linalg.norm(nodes - np.tile(nodes[i,:], [rows, 1]), axis=1)
				if np.sum(np.less(d,5.0)) == 1:
					added_indices.append(i)
					nodes_filtered.append(list(nodes[i,:]))
				else:
					nearby_ind = np.squeeze(np.array(np.nonzero(np.less(d,5.0))))
					added_indices.extend(list(nearby_ind))
					nodes_filtered.append(list(np.average(nodes[nearby_ind,:],axis=0)))
		return np.array(nodes_filtered)	

	def findFrontierNodes(self, nodes, img):
		indices = []
		# Preprocess the image
		free_thresh = self.free_thresh
		img_unseen = np.equal(img, -1) + np.greater(img,free_thresh)*np.less(img, 50.0)
		img_free = np.less(img, free_thresh)*np.greater(img,-0.01)
		img[:] = 0
		img[img_unseen] = 1
		img = img.astype(int)

		plt.cla()
		plt.ion()
		plt.show()

		# Find frontier pixels with a 2D image convolution
		kernel = np.array([[1, 1, 1], [1, -10, 1], [1, 1, 1]])
		frontier_conv = ndimage.convolve(img, kernel, mode='constant', cval=1)
		frontier = np.greater(frontier_conv, 0.1)*img_free

		# Cluster frontiers
		connected_thresh = self.frontier_cluster_thresh
		frontier, num_labels = measure.label(frontier, connectivity=2, return_num=True)
		for label in range(1,num_labels+1):
			cluster = np.equal(frontier, label)
			if (np.sum(cluster) < connected_thresh):
				frontier[cluster] = 0
		frontier = np.greater(frontier,0.1)

		# Subsample frontier image near node locations and if a frontier cluster is nearby, add node index to indices
		node_neighborhood = 30
		img_limits = np.shape(img)
		node_number = 0
		for node in nodes:
			# print(node)
			i_min = max(node[0] - node_neighborhood, 0)
			i_max = min(node[0] + node_neighborhood, img_limits[0])
			j_min = max(node[1] - node_neighborhood, 0)
			j_max = min(node[1] + node_neighborhood, img_limits[1])
			# print("neighborhood index range = [%d : %d, %d : %d]" % (i_min, i_max, j_min, j_max))
			# plt.cla()
			neighborhood = frontier[i_min:i_max, j_min:j_max]
			# plt.imshow(neighborhood, interpolation='nearest')
			# plt.plot(node[1] - j_min, node[0] - i_min, 'bo', markersize=15, markerfacecolor='none')
			# plt.text(node[1]-5 - j_min, node[0]-5 - i_min, str(node_number), fontsize=12, color='xkcd:orange')
			# plt.draw()
			# plt.pause(2.0)
			if np.sum(neighborhood) > 0:
				indices.append(node_number)
			node_number = node_number + 1
		# print("Frontier node indices:")
		# print(indices)

		if self.plot_on and self.plot_steps:
			plt.imshow(frontier, interpolation='nearest')
			plt.plot(nodes[:,1], nodes[:,0], 'bo', markersize=15, markerfacecolor='none')
			plt.plot(nodes[indices,1], nodes[indices,0], 'go', markersize=15, markerfacecolor='none')
			node_label = 0
			for n in nodes:
				plt.text(n[1]-5, n[0]-5, str(node_label), fontsize=12, color='xkcd:orange')
				node_label = node_label + 1
			plt.draw()
			plt.pause(5.0)

		return indices

	def skel2Graph(self, skel):
		# Iterate through each of the nodes in "nodes" and identify the node or node_end that each of their edges connect to
		# Also identify the yaw angle that the edge leaves on
		# Store the edge_paths as well

		# Find nodes by convolving a [[1, 1, 1],[1, 10, 1],[1, 1, 1]] kernel on the skeleton image
		kernel = np.array([[1, 1, 1],[1, 10, 1],[1, 1, 1]])
		skel_conv = ndimage.convolve(skel, kernel, mode='constant', cval=0.0)
		nodes = np.transpose(np.nonzero(np.greater(skel_conv, 12)))
		edges = np.transpose(np.nonzero(np.equal(skel_conv, 12)))
		nodes_end = np.transpose(np.nonzero(np.equal(skel_conv, 11)))

		# Filter out redundant nodes
		nodes = self.filterNodes(nodes)
		if len(nodes):
			nodes_all = np.vstack((nodes, nodes_end))
		else:
			nodes_all = nodes_end
		num_nodes, _ = np.shape(nodes_all)

		rows, cols = np.shape(skel)

		edge_start_indices = []

		# Finds all the starting pixels for the node edges
		for n in nodes:
			# Find the closest connected edge pixels to the node
			node = np.array([int(round(n[0])), int(round(n[1]))])
			neighborhood = skel_conv[(node[0]-4):(node[0]+5), (node[1]-4):(node[1]+5)] # 9x9 in the neighborhood of node
			neigh_rows, neigh_cols = np.shape(neighborhood)
			if not (neigh_rows == 9) or not (neigh_cols == 9):
				continue
			edge_starts = []
			edge_paths = []
			neighbors = []
			for i in range(9):
				for j in range(9):
					# Check if the neighborhood pixel is an edge
					if (neighborhood[i,j] == 12):
						# Check if the edge pixel is adjacent to a node (skel_conv value greater than 12)
						local = node + np.array([i-4, j-4])
						local_neighborhood =  skel_conv[(local[0]-1):(local[0]+2), (local[1]-1):(local[1]+2)]
						# print(local_neighborhood)
						# print(np.greater(local_neighborhood,13))
						if (np.sum(np.greater(local_neighborhood, 12)) > 0):
							# print(local_neighborhood)
							# print(local)
							# print("added to edge list")
							ind = np.ravel_multi_index(local, (rows,cols))
							edge_starts.append(ind)
			# print(edge_starts)
			edge_start_indices.append(edge_starts)

		# print(edge_start_indices)

		neighbors_list = []
		node_id = 0
		edges = []

		# Finds the indices of the neighbor nodes to each node as well as the pixel path of the edges
		for edges_list in edge_start_indices:
			neighbors = []
			edge_id = 1
			for e in edges_list:
				# print("start. node: %d edge: %d" % (node_id, edge_id))
				# Initialize the edge path holder array
				path = []
				edge = dict()
				edge['parent'] = node_id

				# Find the convolution neighborhood around the starting edge pixel
				current_ind = np.unravel_index(e, (rows, cols))
				# print(current_ind)
				neighborhood = np.copy(skel_conv[(current_ind[0]-1):(current_ind[0]+2), (current_ind[1]-1):(current_ind[1]+2)])
				# print(neighborhood)
				neighborhood[1,1] = 0
				# Identify the initial parent, current_pixel, and child
				if np.sum(np.equal(neighborhood, 12)):
					child_local = np.squeeze(np.nonzero(np.equal(neighborhood, 12)))
					# print(child_local)
					child = current_ind - np.array([1,1]) + child_local
					# print(child)
				else:
					continue # candidate edge doesn't go anywhere

				# Add first path point
				path.append([current_ind[0], current_ind[1]])

				while (skel_conv[child[0], child[1]] == 12):
					# print("step")
					parent = current_ind
					parent_local = parent - child + np.array([1,1])
					current_ind = child
					# print(current_ind)
					# print(parent_local)
					neighborhood = np.copy(skel_conv[(current_ind[0]-1):(current_ind[0]+2), (current_ind[1]-1):(current_ind[1]+2)])
					neighborhood[1,1] = 0
					neighborhood[parent_local[0], parent_local[1]] = 0 # Set the parent neighborhood value to 0 so that we head away from the parent node
					# print(neighborhood)

					child_local = np.squeeze(np.nonzero(np.greater(neighborhood, 10)))
					# print(child_local)
					child = current_ind - np.array([1,1]) + child_local
					# print(child)
					path.append([current_ind[0], current_ind[1]])

				# Find the node or node_end that matches to
				dist = np.linalg.norm(nodes_all - np.tile(child, [num_nodes, 1]), axis=1)
				child_id = np.nonzero(np.less(dist, 3.0))
				# print(child_id)
				# print(np.shape(child_id))
				# print(dist)
				# print(np.shape(dist))
				try:
					edge['child'] = child_id[0][0]
					edge['path'] = np.array(path)
					neighbors.append(child_id[0][0])
					# print(edge)
					edges.append(edge)
					edge_id = edge_id + 1
				except:
					continue # Edge doesn't go anywhere

			# print("neighbors = ")
			# print(neighbors)
			neighbors_list.append(neighbors)
			node_id = node_id + 1

		return nodes, nodes_end, edges

	def occGrid2Img(self):
		# Read size and origin parameters
		x_min = self.occupancy_grid_msg.info.origin.position.x
		y_min = self.occupancy_grid_msg.info.origin.position.y
		x_size = self.occupancy_grid_msg.info.width
		y_size = self.occupancy_grid_msg.info.height
		q = self.occupancy_grid_msg.info.origin.orientation
		yaw_origin = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

		# Read the map data
		img = np.array(self.occupancy_grid_msg.data).reshape((self.occupancy_grid_msg.info.height, self.occupancy_grid_msg.info.width))
		# Pad img with self.img_pad worth of unseen pixels
		img = np.pad(img, self.img_pad, 'constant', constant_values=((-1,-1),(-1,-1)))
		# Save raw img for frontier finding
		img_raw = np.transpose(np.copy(img))
		# Preprocess the data
		if self.seen_unseen_filter:
			img = np.transpose(img.astype(float))
			img_unknown = np.less(img, 60)*np.greater(img,40) + np.equal(img, -1)
			img[np.greater(img,-0.01)] = 1.0
			img[img_unknown] = 0.0
		else:
			img[np.equal(img,-1)] = 100
			img = np.transpose(img.astype(float))/100.0
			img = 1.0 - img


		return img, x_min, y_min, x_size, y_size, yaw_origin, img_raw

	def map2graph(self):
		if (self.map_get):
			self.map_get = False
			if (self.mapType == "OccupancyGrid"):
				if (self.time_msgs):
					start_time = time.time()
				img, x_min, y_min, x_size, y_size, yaw_origin, img_raw = self.occGrid2Img()
				if (self.time_msgs):
					print("--- %0.2f ms: Reading Occupancy Grid message ---" % ((time.time() - start_time)*1000.0))
			else:
				return

			if (x_size == 0): # Some error occurred
				rospy.logwarn("Waiting for a map with data")
				return

			# Gaussian blur timing start
			if (self.time_msgs):
				start_time = time.time()
			
			# img = 1.0 - img
			# Occupancy Grid, gaussian blur the image
			img_blur = cv2.GaussianBlur(img, (self.blur_size, self.blur_size), self.blur_sigma, borderType=1)
			# img_blur = 1.0 - img_blur

			# logical threshold the image for skeletonization
			occGrid = np.less(img_blur, self.thresh)

			if (self.time_msgs):
				print("--- %0.2f ms: Gaussian Blur ---" % ((time.time() - start_time)*1000.0))
			# Get unique entries and counts in the last 10 junction counts
			if (self.plot_on and self.plot_steps):
				plt.cla()
				plt.ion()
				plt.show()

				plt.imshow(img, interpolation='nearest')
				plt.draw()
				plt.pause(0.25)

				plt.imshow(img_blur, cmap=plt.cm.gray, interpolation='nearest')
				plt.draw()
				plt.pause(0.25)


			if (self.time_msgs):
				start_time = time.time()
			# plt.imshow(occGrid, cmap=plt.cm.gray, interpolation='nearest')
			# plt.draw()

			# Call thinning c++ function library
			# Convert the occGrid image into a flattened image of type float
			# occGrid_img = occGrid.astype(float)
			# occGrid_img_list = list(occGrid_img.flatten())
			# rows, cols = np.shape(occGrid)
			# if (self.time_msgs):
			# 	start_cpp_time = time.time()
			# skel = cppthin.voronoi_thin(occGrid_img_list, rows, cols, self.thinning_type)
			# if (self.time_msgs):
			# 	print("--- %0.2f ms: Skeletonization (Cpp only) ---" % ((time.time() - start_cpp_time)*1000.0))
			# skel = np.array(skel)
			# skel = skel.reshape((rows, cols))
			# skel = np.less(skel, 0.5)

			# Python's skimage library
			# skel = thin(~occGrid)
			skel = skeletonize(~occGrid)
			skel = skel.astype(np.uint16)
			if (self.time_msgs):
				print("--- %0.2f ms: Skeletonization ---" % ((time.time() - start_time)*1000.0))

			# Turn skeleton into a graph
			if (self.time_msgs):
				start_time = time.time()
			nodes, nodes_end, edges = self.skel2Graph(skel)
			if (self.time_msgs):
				print("--- %0.2f ms: Skeleton to Graph ---" % ((time.time() - start_time)*1000.0))

			# Plot occGrid, skeleton and graph
			# plt.imshow(~skel, cmap=plt.cm.gray, interpolation='nearest')
			# plt.imshow(~occGrid, cmap=plt.cm.gray, interpolation='nearest')

			# Stack all nodes into one array
			if len(nodes):
				nodes_all = np.vstack((nodes, nodes_end))
			else:
				nodes_all = nodes_end

			# Calculate published topics
			num_nodes, _ = np.shape(nodes_all)

			if (num_nodes < 1):
				# Don't need to find the closest node if there are none
				return
			x_ind = int(round((self.position.x - x_min)/self.voxel_size)) + self.img_pad
			y_ind = int(round((self.position.y - y_min)/self.voxel_size)) + self.img_pad
			home_x_ind = int(round((self.home.x - x_min)/self.voxel_size)) + self.img_pad
			home_y_ind = int(round((self.home.y - y_min)/self.voxel_size)) + self.img_pad

			#ROS topic publishing time start
			if (self.time_msgs):
				start_time = time.time()

			# Determine which nodes_end are adjacent to frontiers.  The edges to these nodes are the unexplored edges
			frontier_node_indices = self.findFrontierNodes(nodes_end, img_raw)
			frontier_node_indices = np.array(frontier_node_indices) + len(nodes)

			# Determine which node is the closest
			dist = np.linalg.norm(nodes_all - np.tile(np.array([x_ind, y_ind]), [num_nodes, 1]), axis=1)
			closest_node = np.argmin(dist)

			# Determine which node is the home node
			dist = np.linalg.norm(nodes_all - np.tile(np.array([home_x_ind, home_y_ind]), [num_nodes, 1]), axis=1)
			home_node = np.argmin(dist)

			# Turn nodes and edges into a graph message
			previousCurrentNodeId = self.graph_msg.currentNodeId
			self.graph_msg = Graph()
			self.graph_msg.header.stamp = rospy.Time.now()
			node_num = 0
			node_ids = 1
			msg_ids = []
			# Add all the nodes first
			for node in nodes_all:
				if (node_num in frontier_node_indices) and not (node_num == home_node):
					# Unexplored node
					node_num = node_num + 1
					msg_ids.append(-1)
					continue
				else:
					newNode = Node()
					if (node_num == home_node):
						newNode.id = 0
					else:
						newNode.id = node_ids
						node_ids = node_ids + 1
					msg_ids.append(newNode.id)
					newNode.position.x = (node[0] - self.img_pad)*self.voxel_size + x_min
					newNode.position.y = (node[1] - self.img_pad)*self.voxel_size + y_min
					self.graph_msg.node.append(newNode)
					node_num = node_num + 1

			self.graph_msg.size = node_ids
			# Add in graph edges to message
			for edge in edges:
				parent_id = msg_ids[edge["parent"]]
				path = edge["path"]
				child_id = msg_ids[edge["child"]]
				path_length, _ = np.shape(edge["path"])
				end_index = min(20, path_length-1)
				x_end = edge["path"][end_index,0]
				y_end = edge["path"][end_index,1]
				angle = np.arctan2(y_end - nodes_all[edge["parent"],1], x_end - nodes_all[edge["parent"],0])
				if child_id >= 0:
					# Explored Edge
					self.graph_msg.node[parent_id].nExploredEdge = self.graph_msg.node[parent_id].nExploredEdge + 1
					self.graph_msg.node[parent_id].neighborId.append(child_id)
					self.graph_msg.node[parent_id].edgeCost.append(path_length)
					self.graph_msg.node[parent_id].exploredEdge.append(angle)
					if edge["child"] >= len(nodes):
						x_end = edge["path"][-end_index,0]
						y_end = edge["path"][-end_index,1]
						angle = np.arctan2(y_end - nodes_all[edge["child"],1], x_end - nodes_all[edge["child"],0])
						# Child is a one edge node, add the reverse connection
						self.graph_msg.node[child_id].nExploredEdge = self.graph_msg.node[child_id].nExploredEdge + 1
						self.graph_msg.node[child_id].neighborId.append(parent_id)
						self.graph_msg.node[child_id].edgeCost.append(path_length)
						self.graph_msg.node[child_id].exploredEdge.append(angle)
				else:
					self.graph_msg.node[parent_id].nUnexploredEdge = self.graph_msg.node[parent_id].nUnexploredEdge + 1
					self.graph_msg.node[parent_id].unexploredEdge.append(angle)

			if msg_ids[closest_node] < 0:
				if previousCurrentNodeId < self.graph_msg.size:
					self.graph_msg.currentNodeId = previousCurrentNodeId
				else:
					self.graph_msg.currentNodeId = 0
			else:
				self.graph_msg.currentNodeId = msg_ids[closest_node]

			# Plotting time start
			if (self.time_msgs):
				start_time = time.time()

			frontier_node_indices = frontier_node_indices - len(nodes)

			# Plot node number labels
			if (self.plot_on):
				plt.cla()
				plt.ion()
				plt.show()
				plt.plot(nodes_end[:,1], nodes_end[:,0], 'bo', markersize=15, markerfacecolor='none')
				plt.plot(nodes_end[frontier_node_indices,1], nodes_end[frontier_node_indices,0], 'go', markersize=15, markerfacecolor='none')
				plt.plot(y_ind, x_ind, 'g*')
				plt.title('map2graph')
				plt.imshow(occGrid + skel, cmap=plt.cm.gray, interpolation='nearest')
				if len(nodes):
					plt.plot(nodes[:,1], nodes[:,0], 'ro', markersize=15, markerfacecolor='none')
				node_label = 0
				for n in nodes_all:
					plt.text(n[1]-5, n[0]-5, str(msg_ids[node_label]), fontsize=12, color='xkcd:orange')
					node_label = node_label + 1
				plt.draw()

			if (self.plot_on and not self.plot_steps):
				plt.pause(0.001)
			if (self.plot_on and self.plot_steps):
				plt.pause(0.25)

			if (self.time_msgs):
				print("--- %0.2f ms: Plotting ---" % ((time.time() - start_time)*1000.0))
		return

	def __init__(self):
		# Initialize ROS node and Subscribers
		node_name = 'map2graph'
		rospy.init_node(node_name)
		self.rate = float(rospy.get_param("map2graph/rate", 5.0))

		# Read params
		self.mapType = str(rospy.get_param("map2graph/mapType", "OccupancyGrid"))
		self.voxel_size = float(rospy.get_param("map2graph/resolution", 0.2))
		self.plot_on = bool(rospy.get_param("map2graph/plotting", True))
		self.plot_steps = bool(rospy.get_param("map2graph/plot_steps", False))
		self.time_msgs = bool(rospy.get_param("map2graph/time_msgs", False))
		self.seen_unseen_filter = bool(rospy.get_param("map2graph/seen_unseen_filter", False))
		home_x = float(rospy.get_param("map2graph/home_x", 0.0))
		home_y = float(rospy.get_param("map2graph/home_y", 0.0))
		self.home = Point()
		self.home.x = home_x
		self.home.y = home_y
		map_blur = float(rospy.get_param("map2graph/map_blur", 5.0))
		map_thresh = float(rospy.get_param("map2graph/map_threshold", 0.4))
		self.frontier_cluster_thresh = int(rospy.get_param("map2graph/frontier_cluster_thresh", 20)) # Minimum cluster size to be considered frontier
		self.free_thresh = float(rospy.get_param("map2graph/free_thresh", 49.5)) # Maximum occupancy probability to be considered free by frontier detection

		# Subscribers
		self.position = Point()
		self.position_get_first = False
		rospy.Subscriber('odometry', Odometry, self.getPosition)
		
		self.map_get_first = False
		rospy.Subscriber('occupancy', OccupancyGrid, self.getOccupancyGrid)
		self.occupancy_grid_msg = OccupancyGrid()

		# Initialize Publisher topics
		# pubTopic = 'edge_poses'
		# self.pub_edge_poses = rospy.Publisher(pubTopic, PoseArray, queue_size=1)
		# self.edge_poses = PoseArray()

		pubTopic = 'graph'
		self.pub_graph = rospy.Publisher(pubTopic, Graph, queue_size=1)
		self.graph_msg = Graph()

		# Gaussian Blur and Thresholding params
		self.blur_sigma = map_blur*2.0 # Works well with 0.4 for OccGrid threshold
		self.blur_size = int(2*np.ceil(2*self.blur_sigma) + 1) # Matlab imgaussfilt default
		self.thresh = map_thresh # log-odds thresh

		# Thinning and graph params
		self.thinning_type = "guo_hall_fast"
		# self.thinning_type = "guo_hall"
		# self.thinning_type = "zhang_suen_fast"
		# self.thinning_type = "morph"

		self.img_pad = 4 # pad the img with occupied cells for neighborhood operations

	def start(self):
		rate = rospy.Rate(self.rate)
		# if (self.time_msgs):
		# start_time = time.time()
		while not rospy.is_shutdown():
			rate.sleep()
			# if (self.time_msgs):
			# print("--- %0.2f ms: Full node loop ---" % ((time.time() - start_time)*1000.0))
			# start_time = time.time()
			if (self.map_get_first and self.position_get_first):
				self.map2graph()
				self.pub_graph.publish(self.graph_msg)
			else:
				if (self.map_get_first == False):
					rospy.loginfo("map2graph - Waiting for map message")
				if (self.position_get_first == False):
					rospy.loginfo("map2graph - Waiting for odometry message")
		return

if __name__ == '__main__':
	num_args = len(sys.argv)
	map2graph = node_skeleton()

	try:
		map2graph.start()
	except rospy.ROSInterruptException:
		pass
