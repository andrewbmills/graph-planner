#!/usr/bin/env python
import sys
import numpy as np
import rospy
# import tf
import cv2
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from nav_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from octomap_msgs.msg import Octomap
import sensor_msgs.point_cloud2 as pc2
# My own wrapped thinning function
import sys
# sys.path.insert(0, '../include/')
# import thin_ext as cppthin
from skimage.morphology import skeletonize, thin
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
	# def getTransform(self): # Position subscriber callback function
	# 	(trans,rot) = self.tf_listener.lookupTransform('/world', '/X1/base_link', rospy.Time(0))
	# 	self.position.x = trans[0]
	# 	self.position.y = trans[1]
	# 	self.position.z = trans[2] + 2.0*self.voxel_size
	# 	return

	def getPosition(self, data):
		self.position = data.pose.pose.position
		self.position.z = self.position.z  + 3.0*self.voxel_size
		self.position_get_first = True
		return

	def getCloud(self, data): # PC2 Subscriber
		self.pc2_data = data
		self.cloud_get = True
		self.cloud_get_first = True
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
				edge['child'] = child_id[0][0]
				edge['path'] = np.array(path)
				neighbors.append(child_id[0][0])
				# print(edge)
				edges.append(edge)
				edge_id = edge_id + 1

			# print("neighbors = ")
			# print(neighbors)
			neighbors_list.append(neighbors)
			node_id = node_id + 1

		return nodes, nodes_end, edges

	def cloud2img(self):
		if (self.cloud_get):
			self.cloud_get = False
			if (self.mapType == "Octomap"):
				# Convert pc2 msg to a python list (this is the free cells vis array)
				cloud = pc2.read_points(self.pc2_data, field_names = ("x", "y", "z"), skip_nans=True)
				points = np.array(list(cloud))

			if (self.mapType == "Voxblox"):
				# Convert pc2 msg to a python list (this is the free cells vis array)
				cloud = pc2.read_points(self.pc2_data, field_names = ("x", "y", "z", "intensity"), skip_nans=True)
				points = np.array(list(cloud))

			# Find the x and y extrema
			try:
				x_min = np.amin(points[:,0])
				x_max = np.amax(points[:,0])
				x_size = int(np.round((x_max - x_min)/self.voxel_size)) + 1 + 2*self.img_pad
				y_min = np.amin(points[:,1])
				y_max = np.amax(points[:,1])
				y_size = int(np.round((y_max - y_min)/self.voxel_size)) + 1 + 2*self.img_pad
			except:
				rospy.logwarn("PointCloud2 msg is empty.")
				return

			# Restrict points to just the z_slice neighborhood
			slice_indices = np.logical_and(np.less(points[:,2], self.position.z + self.num_slices*self.voxel_size/2.0), np.greater(points[:,2], self.num_slices*self.position.z - self.voxel_size/2.0))
			points = points[slice_indices,:]

			# Write the slice to an image
			img = np.zeros(shape=[int(x_size), int(y_size)])
			for p in points:
				x_idx = int(np.round((p[0] - x_min)/self.voxel_size)) + self.img_pad
				y_idx = int(np.round((p[1] - y_min)/self.voxel_size)) + self.img_pad
				if (self.mapType == "Octomap"):
					img[x_idx, y_idx] = 1
				if (self.mapType == "Voxblox"):
					img[x_idx, y_idx] = p[3]

			# logical threshold the image for skeletonization
			if (self.mapType == "Octomap"):
				# Gaussian blur the image
				img_blur = cv2.GaussianBlur(img, (self.blur_size, self.blur_size), self.blur_sigma, borderType=1)
			if (self.mapType == "Voxblox"):
				# Gaussian blur the image
				img_blur = cv2.GaussianBlur(img, (self.blur_size, self.blur_size), self.blur_sigma)

			occGrid = np.less(img_blur, self.thresh)

			# Get unique entries and counts in the last 10 junction counts
			plt.cla()
			plt.ion()
			plt.show()

			if (self.plot_on and self.plot_steps):
				plt.imshow(img, interpolation='nearest')
				plt.draw()
				plt.pause(0.25)

				plt.imshow(img_blur, cmap=plt.cm.gray, interpolation='nearest')
				plt.draw()
				plt.pause(0.25)

			# plt.imshow(occGrid, cmap=plt.cm.gray, interpolation='nearest')
			# plt.draw()

			# Call thinning c++ function library
			# Convert the occGrid image into a flattened image of type float
			# occGrid_img = occGrid.astype(float)
			# occGrid_img_list = list(occGrid_img.flatten())
			# rows, cols = np.shape(occGrid)
			# skel = cppthin.voronoi_thin(occGrid_img_list, rows, cols, self.thinning_type)
			# skel = np.array(skel)
			# skel = skel.reshape((rows, cols))
			# skel = np.less(skel, 0.5)

			# Python's skimage library
			# skel = skeletonize(~occGrid)
			skel = thin(~occGrid)
			skel = skel.astype(np.uint16)

			nodes, nodes_end, edges = self.skel2Graph(skel)

			# Plot occGrid, skeleton and graph
			# plt.imshow(~skel, cmap=plt.cm.gray, interpolation='nearest')
			# plt.imshow(~occGrid, cmap=plt.cm.gray, interpolation='nearest')
			plt.plot(nodes_end[:,1], nodes_end[:,0], 'go', markersize=15, markerfacecolor='none')
			plt.title('map2graph')

			# Plot node number labels
			if (self.plot_on):
				plt.imshow(occGrid + skel, cmap=plt.cm.gray, interpolation='nearest')
				if len(nodes):
					nodes_all = np.vstack((nodes, nodes_end))
					plt.plot(nodes[:,1], nodes[:,0], 'ro', markersize=15, markerfacecolor='none')
				else:
					nodes_all = nodes_end
				node_label = 0
				for n in nodes_all:
					plt.text(n[1]-5, n[0]-5, str(node_label), fontsize=12)
					node_label = node_label + 1
				plt.draw()

			if (self.plot_on and not self.plot_steps):
				plt.pause(0.001)
			if (self.plot_on and self.plot_steps):
				plt.pause(0.25)

			# Calculate published topics
			num_nodes, _ = np.shape(nodes_all)

			if (num_nodes < 1):
				# Don't need to find the closest node if there are none
				return

			x_ind = int(round((self.position.x - x_min)/self.voxel_size)) + self.img_pad
			y_ind = int(round((self.position.y - y_min)/self.voxel_size)) + self.img_pad
			dist = np.linalg.norm(nodes_all - np.tile(np.array([x_ind, y_ind]), [num_nodes, 1]), axis=1)
			# print(dist)
			closest_node_id = np.argmin(dist)
			self.closest_node.point.x = (nodes_all[closest_node_id,0] - self.img_pad)*self.voxel_size + x_min
			self.closest_node.point.y = (nodes_all[closest_node_id,1] - self.img_pad)*self.voxel_size + y_min
			self.closest_node.point.z = self.position.z
			# print(self.closest_node)

			# Edge angle calculations
			edge_angles = []
			self.edge_poses = PoseArray()
			self.edge_poses.header.frame_id = "world"
			self.edge_poses.header.stamp = rospy.Time.now()
			# print(closest_node_id)
			for i in range(len(edges)):
				# print(edges[i]["parent"])
				if (edges[i]["parent"] == closest_node_id):
					path_length, _ = np.shape(edges[i]["path"])
					end_index = min(20, path_length-1)
					x_end = edges[i]["path"][end_index,0]
					y_end = edges[i]["path"][end_index,1]
					angle = np.arctan2(y_end - nodes_all[closest_node_id,1], x_end - nodes_all[closest_node_id,0])
					edge_angles.append(angle)
					new_pose = Pose()
					new_pose.position.x = (nodes_all[closest_node_id,0] - self.img_pad)*self.voxel_size + x_min
					new_pose.position.y = (nodes_all[closest_node_id,1] - self.img_pad)*self.voxel_size + y_min
					new_pose.position.z = self.position.z
					q = euler2quaternion(np.array([angle, 0, 0]))
					new_pose.orientation.w = q[0]
					new_pose.orientation.x = q[1]
					new_pose.orientation.y = q[2]
					new_pose.orientation.z = q[3]
					self.edge_poses.poses.append(new_pose)

			self.edge_list.data = edge_angles
			dim = MultiArrayDimension()
			dim.size = len(edge_angles)
			dim.label = "edge_angles"
			dim.stride = int(32)
			self.edge_list.layout.dim[0] = dim
		return

	def __init__(self):
		# Initialize ROS node and Subscribers
		node_name = 'node_skeleton'
		rospy.init_node(node_name)
		self.rate = float(rospy.get_param("/map2graph/rate", 5.0))

		# Read params to see if the map is an OctoMap or a Voxblox
		# self.mapType = "Voxblox"
		self.mapType = str(rospy.get_param("/map2graph/mapType", "Voxblox"))
		self.voxel_size = float(rospy.get_param("/map2graph/resolution", 0.2))
		self.num_slices = int(rospy.get_param("/map2graph/num_slices", 1))
		self.plot_on = bool(rospy.get_param("/map2graph/plotting", True))
		self.plot_steps = bool(rospy.get_param("/map2graph/plot_steps", False))
		map_blur = float(rospy.get_param("/map2graph/map_blur", 5.0))
		map_thresh = float(rospy.get_param("/map2graph/map_threshold", 0.4))

		# Subscribers
		# rospy.Subscriber('X1/odometry', Odometry, self.getPosition)
		rospy.Subscriber('/odometry', Odometry, self.getPosition)
		self.position_get_first = False
		# rospy.Subscriber('X1/voxblox_node/esdf_pointcloud', PointCloud2, self.getCloud)
		rospy.Subscriber('/pointcloud', PointCloud2, self.getCloud)
		self.cloud_get = False
		self.cloud_get_first = False

		# Initialize Publisher topics
		pubTopic = 'node_skeleton/closest_node'
		self.pub_closest_node = rospy.Publisher(pubTopic, PointStamped, queue_size=10)
		self.closest_node = PointStamped()
		self.closest_node.header.frame_id = "world"

		pubTopic = 'node_skeleton/closest_edges'
		self.pub_closest_edges = rospy.Publisher(pubTopic, Float32MultiArray, queue_size=10)
		self.edge_list = Float32MultiArray()
		dim = MultiArrayDimension()
		dim.size = 0
		dim.label = "edge_angles"
		dim.stride = int(32)
		self.edge_list.layout.dim.append(dim)

		pubTopic = 'node_skeleton/closest_node_poses'
		self.pub_edge_poses = rospy.Publisher(pubTopic, PoseArray, queue_size=10)
		self.edge_poses = PoseArray()

		# Initialize tf listener and holding object
		# self.tf_listener = tf.TransformListener()
		self.position = Point()

		# Gaussian Blur and Thresholding params
		if self.mapType == "Voxblox":
			self.blur_sigma = map_blur # Works well with 0.4 ESDF threshold
			self.blur_size = int(2*np.ceil(2*self.blur_sigma) + 1) # Matlab imgaussfilt default
			self.thresh = map_thresh
		else:
			self.blur_sigma = map_blur*2.0 # Works well with 0.4 for OccGrid threshold
			self.blur_size = int(2*np.ceil(2*self.blur_sigma) + 1) # Matlab imgaussfilt default
			self.thresh = map_thresh # log-odds thresh


		# Thinning and graph params
		# self.thinning_type = "guo_hall_fast"
		self.thinning_type = "guo_hall"
		# self.thinning_type = "zhang_suen_fast"
		# self.thinning_type = "morph"

		self.img_pad = 4 # pad the img with occupied cells for neighborhood operations

	def start(self):
		rate = rospy.Rate(self.rate)
		while not rospy.is_shutdown():
			rate.sleep()
			# self.getTransform()
			if (self.cloud_get_first and self.position_get_first):
				self.cloud2img()
			else:
				if (self.cloud_get_first == False):
					rospy.loginfo("map2graph - Waiting for map Pointcloud2 message")
				if (self.position_get_first == False):
					rospy.loginfo("map2graph - Waiting for odometry message")

			self.pub_closest_node.publish(self.closest_node)
			self.pub_closest_edges.publish(self.edge_list)
			self.pub_edge_poses.publish(self.edge_poses)
		return

if __name__ == '__main__':
	num_args = len(sys.argv)
	map2graph = node_skeleton()

	try:
		map2graph.start()
	except rospy.ROSInterruptException:
		pass
