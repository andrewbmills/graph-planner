// Python 2.7
// g++ -I /usr/include/python2.7 -fpic -c -o thin_ext.o thin_ext.cpp `pkg-config --libs --cflags opencv`
// g++ -o thin_ext.so -shared thin_ext.o -lboost_python -lpython2.7 `pkg-config --libs --cflags opencv`
// Python 3.6
// g++ -I /usr/include/python3.6 -fpic -c -o thin_ext.o thin_ext.cpp `pkg-config --libs --cflags opencv`
// g++ -o thin_ext.so -shared thin_ext.o -lboost_python3 -lpython3.6m `pkg-config --libs --cflags opencv`

#include <boost/python.hpp>
#include "voronoi.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace boost::python;

list voronoi_thin(list img_list, const int rows, const int cols, str implementation_name_str)
{
	char const* implementation_name = extract<char const*>(implementation_name_str);

	// Copy img data into a cpp array
	unsigned char img_array[rows*cols];
	for (int i = 0; i < len(img_list); ++i){
		if (extract<float>(img_list[i]) >= 0.5){
			img_array[i] = (unsigned char)(0);
		} else {
			img_array[i] = (unsigned char)(255);
		}
	}

	// Copy cpp array data to Mat1b object
	cv::Mat1b img(rows, cols);
	std::memcpy(img.data, img_array, rows*cols*sizeof(unsigned char));

	// Initialize thinner
	VoronoiThinner thinner;

	// Thin the image
	thinner.thin(img, implementation_name, true);

	// Show images
	// cv::imshow("query", img);
	// cv::imshow(implementation_name, thinner.get_skeleton());
	// cv::waitKey(0);

	// Return image by copying to skel array pointer
	cv::Mat1b skeleton_mat = thinner.get_skeleton();
	list skel;
	unsigned char skel_array[rows*cols];
	std::memcpy(skel_array, skeleton_mat.data, rows*cols*sizeof(unsigned char));
	for (int i = 0; i < len(img_list); ++i){
		if (((int)skel_array[i]) >= 127){
			skel.append(0.0);
		} else {
			skel.append(1.0);
		}
	}

	return skel;
}

BOOST_PYTHON_MODULE(thin_ext)
{
	def("voronoi_thin", voronoi_thin);
}