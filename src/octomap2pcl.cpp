// g++ octomap2pcl.cpp -g -o octomap2pcl.o -I /opt/ros/melodic/include -I /usr/include/c++/7.3.0 -I /home/andrew/catkin_ws/devel/include -I /home/andrew/catkin_ws/src/octomap_msgs/include -I /usr/include/pcl-1.8 -I /usr/include/eigen3 -L /usr/lib/x86_64-linux-gnu -L /home/andrew/catkin_ws/devel/lib -L /opt/ros/melodic/lib -Wl,-rpath,opt/ros/melodic/lib -lroscpp -lrosconsole -lrostime -lroscpp_serialization -loctomap -lboost_system -lpcl_common -lpcl_io

#include <math.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

// global variables
sensor_msgs::PointCloud2 freeCellsMsg;

void callbackOctomap(const octomap_msgs::Octomap::ConstPtr msg)
{
  ROS_INFO("Getting OctoMap message...");
  octomap::OcTree* mytree = new octomap::OcTree(msg->resolution);
  mytree = (octomap::OcTree*)octomap_msgs::binaryMsgToMap(*msg);
  ROS_INFO("AbstractOcTree cast into OcTree.");

  ROS_INFO("Parsing Octomap...");
  // Make sure the tree is at the same resolution as the esdf we're creating.  If it's not, change the resolution.
  ROS_INFO("Tree resolution is %f meters.", mytree->getResolution());

  // Loop through tree and extract occupancy info into data
  double size, value;
  float lower_corner[3];
  int idx, depth, width;
  int lowest_depth = (int)mytree->getTreeDepth();
  int count = 0;
  int freeCount = 0;
  int occCount = 0;
  float voxel_size = (float)mytree->getResolution();
  // PointCloud2 point and msg objects
  pcl::PointXYZ point;
  pcl::PointCloud<pcl::PointXYZ>::Ptr freeCells(new pcl::PointCloud<pcl::PointXYZ>);

  ROS_INFO("Starting tree iterator on OcTree with max depth %d", lowest_depth);
  for(octomap::OcTree::leaf_iterator it = mytree->begin_leafs(),
       end=mytree->end_leafs(); it!=end; ++it)
  {
    // Get data from node
    depth = (int)it.getDepth();
    point.x = (float)it.getX();
    point.y = (float)it.getY();
    point.z = (float)it.getZ();
    size = it.getSize();

    if (it->getValue() > 0) {
      value = false;
      occCount++;
    } else {
      value = true;
      freeCount++;
      // Put data into PC2
      if (depth == lowest_depth){
        if (value) {
          freeCells->points.push_back(point);
        }
      } else{ // Fill in all the voxels internal to the leaf
        width = (int)std::pow(2.0, (double)(lowest_depth-depth));
        lower_corner[0] = point.x - size/2.0 + voxel_size/2.0;
        lower_corner[1] = point.y - size/2.0 + voxel_size/2.0;
        lower_corner[2] = point.z - size/2.0 + voxel_size/2.0;

        for (int i=0; i<width; i++){
          point.x = lower_corner[0] + i*voxel_size;
          for (int j=0; j<width; j++){
            point.y = lower_corner[1] + j*voxel_size;
            for (int k=0; k<width; k++){
              point.z = lower_corner[2] + k*voxel_size;
              if (value) {
                freeCells->points.push_back(point);
              }
            }
          }
        }
      }
    }
  }

  pcl::toROSMsg(*freeCells, freeCellsMsg);

  // Free memory
  delete mytree;

  ROS_INFO("Octomap message received.  %d leaves labeled as occupied.  %d leaves labeled as free.", occCount, freeCount);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "octomap2pcl");
  ros::NodeHandle n;

  // Declare subscriber and publisher
  ros::Subscriber sub = n.subscribe("/octomap_binary", 1, callbackOctomap);
  ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2>("octomap_free_cells_all", 5);

  // Update Rate
  ros::Rate r(10.0);
  int sequence = 1;

  while (ros::ok())
  {
    r.sleep();
    ros::spinOnce();
    freeCellsMsg.header.seq = sequence;
    freeCellsMsg.header.frame_id = "world";
    freeCellsMsg.header.stamp = ros::Time();
    pub.publish(freeCellsMsg);
    sequence++;
  }

  return 0;
}