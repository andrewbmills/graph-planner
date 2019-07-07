#!/usr/bin/env python
import sys
import numpy as np
import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def cartesian2spherical(xyz):
	# Converts a numpy array of N (x,y,z) cartesian coordinates (N x 3) to spherical coordinates
	xy = xyz[:,0]**2 + xyz[:,1]**2 # Radial distance squared in xy plane
	r = np.sqrt(xy + xyz[:,2]**2) # Radial distance
	elev = np.arctan2(xyz[:,2], np.sqrt(xy)) # elevation measured from xy-plane (rad)
	az = np.arctan2(xyz[:,1], xyz[:,0]) # azimuth (rad)
	return np.transpose(np.vstack((r, elev, az)))

class wfi_centering_controller:
	def getCloud(self, data): # PC2 Subscriber
		self.pc2_data = data
		self.cloud_get = True
		self.cloud_get_first = True
		return

	def wide_field_integration(self):
		if (self.cloud_get):
			self.cloud_get = False

			# Convert pc2 msg to a python list
			points = []
			for p in pc2.read_points(self.pc2_data, field_names = ("x", "y", "z"), skip_nans=True):
				points.append([p[0], p[1], p[2]])
			points = np.array(points)

			# Convert depth cloud to spherical coordinates and take only the points with the smallest magnitude elevation
			points_spherical = cartesian2spherical(points)

			# Filter points based on elevation
			self.points_filtered = np.empty([0,3])
			for p in points_spherical:
				if (p[1] < (0.04)) and (p[1] > (0.0)):
					self.points_filtered = np.vstack((self.points_filtered, [p[0], p[1], p[2]]))

		# Extract depth and azimuth from filtered PointClouds
		d = self.points_filtered[:,0]
		az = self.points_filtered[:,2]

		# Calculate the nearness
		near = 1/d

		if (self.show_plot):
			plt.cla()
			plt.ion()
			plt.show()
			plt.plot(az, d)
			plt.title('Centering Depth vs azimuth')
			plt.pause(0.001)

		# Calculate the dot products with sin(az) and sin(2*az)
		z1 = np.inner(np.sin(az), near) # Lateral position error (dy)
		z2 = np.inner(np.sin(2.0*az), near) # yaw angle error (dpsi)

		# Calculate the twist command
		self.cmd_vel.linear.x = self.u0
		self.cmd_vel.angular.z = -0.1*(self.K1*z1 + self.K2*z2)
		return

	def __init__(self):
		# Set controller specific parameters
		self.u0 = float(rospy.get_param("/centering/speed", 0.8)) # m/s
		self.rate = float(rospy.get_param("/centering/rate", 10.0)) # Hz
		self.show_plot = bool(rospy.get_param("/centering/plot", False)) # Bool

		# Initialize ROS node and Subscribers
		node_name = 'centering_controller'
		rospy.init_node(node_name)
		rospy.Subscriber('points', PointCloud2, self.getCloud)
		self.points = np.empty([0,3])
		self.cloud_get_first = False

		# Initialize Publisher topic(s)
		self.pub1 = rospy.Publisher('cmd_vel', Twist, queue_size=10)

		# Initialize twist object for publishing
		self.cmd_vel = Twist()

		# Controller tuning
		self.K1 = 0.0141421
		self.K2 = 0.0206464
		self.yaw_rate_max = float(rospy.get_param("yaw_rate_max", 0.2))

	def start(self):
		rate = rospy.Rate(self.rate)
		while not rospy.is_shutdown():
			rate.sleep()
			if (self.cloud_get_first):
				self.wide_field_integration()
			# Saturate the turn rate command
			if (np.absolute(self.cmd_vel.angular.z) >= self.yaw_rate_max):
				self.cmd_vel.angular.z = np.sign(self.cmd_vel.angular.z)*self.yaw_rate_max
			self.pub1.publish(self.cmd_vel)
		return

if __name__ == '__main__':
	num_args = len(sys.argv)
	controller = wfi_centering_controller()

	try:
		controller.start()
	except rospy.ROSInterruptException:
		pass
