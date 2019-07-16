#!/usr/bin/env python
import sys
import numpy as np
import rospy
import matplotlib.pyplot as plt
from scipy import signal
from std_msgs.msg import Int32
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def angleDiff(a, b):
	# Computes a-b, preserving the correct sign (counter-clockwise positive angles)
	# All angles are in degrees
	a = (360000 + a) % 360
	b = (360000 + b) % 360
	d = a - b
	d = (d + 180) % 360 - 180
	return d

def cartesian2spherical(xyz):
	# Converts a numpy array of N (x,y,z) cartesian coordinates (N x 3) to spherical coordinates
	xy = xyz[:,0]**2 + xyz[:,1]**2 # Radial distance squared in xy plane
	r = np.sqrt(xy + xyz[:,2]**2) # Radial distance
	elev = np.arctan2(xyz[:,2], np.sqrt(xy)) # elevation measured from xy-plane (rad)
	az = np.arctan2(xyz[:,1], xyz[:,0]) # azimuth (rad)
	return np.transpose(np.vstack((r, elev, az)))

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

class wfi_centering_controller:
	def getCloud(self, data): # PC2 Subscriber
		self.pc2_data = data
		self.robot_frame = data.header.frame_id
		self.cloud_get = True
		self.cloud_get_first = True
		return

	def detectJunctions(self):
		peaks, _ = signal.find_peaks(self.d, height=self.min_peak_height, distance=(int(np.round(len(self.d)/7.0))))
		# print(peaks)
		self.junction_headings = []
		self.junction_directions = PoseArray()
		self.junction_directions.header.frame_id = self.robot_frame
		
		if len(peaks) > 0:
			if len(peaks) > 1:
				# Check to see if the first and last peaks are the same
				bookend_distance = float(peaks[0] + len(self.az)-peaks[-1])
				# print("Checking if the first and last peaks are the same")
				# print("first peak index = %d" %(peaks[0]))
				# print("last peak index = %d" %(peaks[-1]))
				# print("Azimuth array length = %d" % (len(self.az)))
				# print("bookend distance = %0.0f" % (bookend_distance))
				if (bookend_distance/(float(len(self.az))) < 0.1):
					peaks[0] = (peaks[0] + peaks[-1]) % len(self.az)
					peaks = peaks[:-1]

			self.junction_count.data = len(peaks)
			# Publish and plot
			for peak_id in peaks:
				peak_pose = Pose()
				peak_pose.orientation = euler2quaternion(np.array([self.az[peak_id], 0.0, 0.0]))
				self.junction_directions.poses.append(peak_pose)
				self.junction_headings.append(self.az[peak_id])
			if (self.show_plot):
				plt.cla()
				plt.ion()
				plt.show()
				plt.plot(self.az, self.d)
				plt.plot(self.az[peaks], self.d[peaks], 'o')
				plt.title('Centering Depth vs azimuth with peaks')
				plt.pause(0.001)
		else:
			self.junction_count.data = 0
			plt.cla()
			plt.ion()
			plt.show()
			plt.plot(self.az, self.d)
			plt.title('Centering Depth vs azimuth with peaks')
			plt.pause(0.001)
		return


	def wideFieldIntegration(self):
		if (self.cloud_get):
			self.cloud_get = False

			# Convert pc2 msg to a python list
			cloud = pc2.read_points(self.pc2_data, field_names = ("x", "y", "z"), skip_nans=True)
			points = np.asarray(list(cloud))

			# Convert depth cloud to spherical coordinates and take only the points with the smallest magnitude elevation
			points_spherical = cartesian2spherical(points)

			# Take the max depth at the current azimuth (n_elev points at each azimuth ordered in ascending azimuth)
			try:
				n_elev = 15
				num_points, _ = np.shape(points_spherical)
				# print(num_points)
				self.points_filtered = np.empty([int(np.floor(num_points/n_elev)),2])
				for i in range(int(np.floor(num_points/n_elev))):
					# print(points_spherical[(n_elev*i):(n_elev*i+n_elev),2])
					self.points_filtered[i,0] = np.amax(points_spherical[(n_elev*i):(n_elev*i+n_elev),0])
					self.points_filtered[i,1] = points_spherical[n_elev*i,2]
			except:
				return

			# self.points_filtered = np.empty([0,3])
			# for p in points_spherical:
			# 	print(p)
			# 	if (p[1] < (0.04)) and (p[1] > (0.0)):
			# 		self.points_filtered = np.vstack((self.points_filtered, [p[0], p[1], p[2]]))

		# Extract depth and azimuth from filtered PointClouds
		self.d = self.points_filtered[:,0]
		self.az = self.points_filtered[:,1]

		# Calculate the nearness
		near = 1/self.d

		# if (self.show_plot):
		# 	plt.cla()
		# 	plt.ion()
		# 	plt.show()
		# 	plt.plot(self.az, self.d)
		# 	plt.title('Centering Depth vs azimuth')
		# 	plt.pause(0.001)

		# Calculate the dot products with sin(az) and sin(2*az)
		z1 = np.inner(np.sin(self.az), near) # Lateral position error (dy)
		z2 = np.inner(np.sin(2.0*self.az), near) # yaw angle error (dpsi)

		# Calculate the twist command
		self.cmd_vel.linear.x = self.u0
		self.cmd_vel.angular.z = -0.1*(self.K1*z1 + self.K2*z2)
		return

	def __init__(self):
		# Set controller specific parameters
		self.u0 = float(rospy.get_param("/centering/speed", 0.8)) # m/s
		self.rate = float(rospy.get_param("/centering/rate", 10.0)) # Hz
		self.show_plot = bool(rospy.get_param("/centering/plot", False)) # Bool
		self.min_peak_height = float(rospy.get_param("/centering/min_peak_height", 10.0)) # meters

		# Initialize ROS node and Subscribers
		node_name = 'centering_controller'
		rospy.init_node(node_name)
		rospy.Subscriber('points', PointCloud2, self.getCloud)
		self.points = np.empty([0,3])
		self.cloud_get_first = False

		# Initialize Publisher topic(s)
		self.pub1 = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.pub2 = rospy.Publisher('junction_count', Int32, queue_size=10)
		self.pub3 = rospy.Publisher('junction_directions', PoseArray, queue_size=10)

		# Initialize twist object for publishing
		self.cmd_vel = Twist()

		# Junction count property for publishing
		self.junction_count = Int32()
		self.junction_count.data = 0
		self.junction_directions = PoseArray()

		# Controller tuning
		self.K1 = 0.0141421
		self.K2 = 0.0206464
		self.Kspeed = 0.01
		self.yaw_rate_max = float(rospy.get_param("yaw_rate_max", 0.2))
		return

	def start(self):
		rate = rospy.Rate(self.rate)
		loops = 0
		while not rospy.is_shutdown():
			rate.sleep()
			if (self.cloud_get_first):
				self.wideFieldIntegration()
				# Run a junction detector on a fifth of the loops
				if not (loops % 20):
					self.detectJunctions()
				loops = loops + 1
				# Turn around at dead ends
				if self.junction_count.data == 1:
					# Turn until you're facing the only junction_direction
					if abs(self.junction_headings[0]) > 45*np.pi/180:
						print("Dead end detected.  Turning around.")
						self.cmd_vel.angular.z = np.sign(self.junction_headings[0])*self.yaw_rate_max
						print("Yaw rate = %0.2f deg/s" % (self.cmd_vel.angular.z*(180/np.pi)))
						self.cmd_vel.linear.x = -self.u0/5.0

			# Saturate the turn rate command
			if (np.absolute(self.cmd_vel.angular.z) >= self.yaw_rate_max):
				self.cmd_vel.angular.z = np.sign(self.cmd_vel.angular.z)*self.yaw_rate_max
			self.pub1.publish(self.cmd_vel)
			self.pub2.publish(self.junction_count)
			self.pub3.publish(self.junction_directions)
		return

if __name__ == '__main__':
	num_args = len(sys.argv)
	controller = wfi_centering_controller()

	try:
		controller.start()
	except rospy.ROSInterruptException:
		pass
