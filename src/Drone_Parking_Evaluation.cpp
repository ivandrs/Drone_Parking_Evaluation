//============================================================================
// Name        : Drone_Parking_Evaluation.cpp
// Author      : ivan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>       /* sqrt */
#include <utility>      // std::pair
#include <fstream>
/*
 * cmake -DOPENCV_EXTRA_MODULES_PATH=home/ivan/Downloads/opencv_contrib-master/modules -DBUILD_EXAMPLES=ON /home/ivan/EXDropbox/MasterThesis/opencv-2.4.13.6/


 * https://github.com/opencv/opencv_contrib
 * https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
 * https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_eclipse/linux_eclipse.html
 * https://docs.opencv.org/master/df/d2d/group__ximgproc.html
 * https://github.com/opencv/opencv/issues/7578
 * -std=c++11 to the CXX_FLAG? GCC version? g++ version?
 */


void ShowAndSave(const std::string& windowName, const int cols, const int rows,
		const cv::Mat& Img, const std::string& baseDir, bool wait) {
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, cols / 2, rows / 4);
	imshow(windowName, Img);
	imwrite(baseDir + windowName + ".png", Img);
	if (wait)
		cv::waitKey(0);
}




/// Global variables

int threshold_value = 0;
int threshold_type = 3;;


int const max_BINARY_value = 255;

cv::Mat src, src_gray, dst;




std::string window_name;
/// Function headers
//void Threshold_Demo( int, void* );

/**
 * @function Threshold_Demo
 */
void Threshold_Demo( int, void*)
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

  cv::imshow( window_name, dst );
}

//threshold Canny
int lowThresholdG = 30;
int ratioG = 3;
int kernel_sizeG = 2;
void Threshold_Canny( int, void*)
{

	if (kernel_sizeG<3)
		kernel_sizeG=3;

	kernel_sizeG= 2 * ( (int)(kernel_sizeG / 2.0f) ) + 1;
cv::Canny(  src_gray, dst, lowThresholdG, lowThresholdG*ratioG, kernel_sizeG );
cv::imshow( window_name, dst );


}



void getAngle(const std::vector<cv::Vec4i>& line,
		std::vector<cv::Vec4i>& straighLines,
		double maxAngle = 100 ) {
	std::vector<double> angles;

	for (size_t i = 0; i < line.size(); i++) {
		cv::Vec4i l = line[i];
		// draw the lines
		cv::Point p1, p2;
		p1 = cv::Point(l[0], l[1]);
		p2 = cv::Point(l[2], l[3]);
		//calculate angle in radian,  if you need it in degrees just do angle * 180 / PI
		float angle = atan2(p1.y - p2.y, p1.x - p2.x) * 180 / CV_PI;
		angles.push_back(angle);
		if (angle > maxAngle || angle < -maxAngle)
			straighLines.push_back(l);
	}
}


struct DistanceLines
{
    double distance;
    cv::Vec4i line;

    DistanceLines(double d, const cv::Vec4i & l) : distance(d), line(l) {}

    bool operator < (const DistanceLines& dl) const
    {
        return (distance > dl.distance);
    }
};

//threshold HoughLinesP
int rho=10;
int theta= 1;
int threshold=50;//10000;
int minLineLength=50;//3000;
int maxLineGap=50;//700;

void plotLines(const std::vector<cv::Vec4i>& lines, cv::Mat copy_src, cv::Scalar color ) {
	for (size_t i = 0; i < lines.size(); i++) {
		cv::Vec4i l = lines[i];
		cv::line(copy_src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
				color, 10, CV_AA);
	}
	cv::imshow(window_name, copy_src);
}

void plotLines(const std::vector<DistanceLines> & dlines, cv::Mat copy_src, cv::Scalar color ) {
	std::vector<cv::Vec4i> lines;
	for (size_t i = 0; i < dlines.size(); i++) {
		lines.push_back(dlines[i].line);
	}
	plotLines(lines, copy_src, color);
}

double euclidDist(const cv::Vec4i& line)
{
    return sqrt(static_cast<double>((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3])));
}




void Threshold_HoughLinesP( int, void*)
{
	double thetarad= theta/10*CV_PI/180;
	thetarad=CV_PI/180;
	double drho =rho/10;
	drho=1;


	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(src_gray, lines, drho, thetarad, threshold, minLineLength, maxLineGap );

	cv::Mat copy_src;
	cv::cvtColor(src_gray, copy_src, cv::COLOR_GRAY2BGR);
	//cv::Mat copy_src=src_gray.clone();

	plotLines(lines, copy_src, cv::Scalar(255, 255, 255));
	std::vector<cv::Vec4i> straighLines;
	getAngle(lines, straighLines, 175);
	plotLines(straighLines, copy_src, cv::Scalar(100, 100, 255));

	// get longer lines
	size_t numlines =5;


	std::vector<DistanceLines> longerLines;
	std::vector<DistanceLines> linesWLenghts;

	for (cv::Vec4i line : lines)
	{
		linesWLenghts.push_back(DistanceLines(euclidDist(line),line));
	}
	std::sort(linesWLenghts.begin(), linesWLenghts.end());

	for (size_t i=0; i<numlines;i++)
	{
		if (lines.size()>numlines)
		{
			longerLines.push_back(linesWLenghts[i]);
		}
	}
	plotLines(longerLines, copy_src, cv::Scalar(100, 200, 255));


}

//using namespace cv;s
//using namespace std;
/*int main( int argc, char** argv )
{
  /// Load an image
  src = imread( argv[1], 1 );*/



int main()
{
	std::cout << "!!!start Drone_Parking_Evaluation 2!!!" << std::endl;

	const std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_fern.JPG";
	const std::string baseDir="output/";
	cv::Mat raw=cv::imread( rawImgAdd  );
	if (raw.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	const int cols= raw.cols;
	const int rows= raw.rows;

	//original image
	//ShowAndSave("Raw", cols, rows, raw, baseDir,0);



/*
	//1 (RoI)cut image, kept below default horizon =
	// TODO: CHECK. a perspective Region of interest can be applied from here too
	//TODO: is not necessary to crop, i can do all the operations over the roi and i can save the cuting

	int highp=760;
	cv::Rect inROI(0,highp,cols,rows-highp);
	cv::Point topRight(1220,highp);
	cv::Point topLeft(700,highp);
	cv::Point bottomRight(1920,1208);
	cv::Point bottomLeft(1,1208);
	cv::Mat mask = cv::Mat::zeros(raw.size(), raw.type());
	cv::Mat dstImage = cv::Mat::zeros(raw.size(), raw.type());
	std::vector<std::vector<cv::Point> > vpts;
	std::vector<cv::Point> pts;
	pts.push_back(topRight);
	pts.push_back(topLeft);
	pts.push_back(bottomLeft);
	pts.push_back(bottomRight);

	vpts.push_back(pts);
	cv::fillPoly(mask,vpts, cv::Scalar(255, 255, 255), 8, 0);
	raw.copyTo(dstImage, mask);

	//PLOT: Cropped dstImage

	windowName = "dstImage";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/4);
	imshow(windowName, dstImage);
	imwrite( baseDir+windowName+".png", dstImage );
	cv::waitKey(0);

	cv::Mat roiRaw= dstImage(inROI);
	//cv::Mat roiRaw= raw(inROI);
	//roiRaw=dstImage;
	//PLOT: Cropped image

	windowName = "Rawroi";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/4);
	imshow(windowName, roiRaw);
	imwrite( baseDir+windowName+".png", roiRaw );
	cv::waitKey(0);

*/

	//2 RGB2Gray
	cv::Mat imgGray;
	cvtColor(raw, imgGray, cv::COLOR_BGR2GRAY);
	//ShowAndSave("imgGray", cols, rows, imgGray, baseDir,0);

	//3 Gausian filter
	cv::Mat imgGauss;
	GaussianBlur( imgGray, imgGauss, cv::Size( 15, 15 ), 3, 0 ); //TODO: try size 5 sugested by open cv
	//ShowAndSave("imgGauss", cols, rows, imgGauss, baseDir,0);




	//Applying threshold to only work with black an white constranst

if (false){
	window_name = "Threshold Demo";
	src_gray=imgGray;

	  /// Create a window to display results
	cv::namedWindow( window_name, cv::WINDOW_NORMAL );
	cv::resizeWindow(window_name, cols / 2, rows / 4);
	int const max_type = 4;

	  /// Create Trackbar to choose type of Threshold
	std::string trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";

	  cv::createTrackbar( trackbar_type,
	                  window_name, &threshold_type,
	                  max_type, Threshold_Demo);
	  std::string trackbar_value = "Value";
	  int const max_value = 255;
	  cv::createTrackbar( trackbar_value,
	                  window_name, &threshold_value,
	                  max_value, Threshold_Demo );

	  /// Call the function to initialize
	  Threshold_Demo( 0, 0 );

	  /// Wait until user finishes program
	  while(true)
	  {
	    int c;
	    c = cv::waitKey( 20 );
	    if( (char)c == 27 )
	      { break; }
	   }


}




	//4 Edge detection
	//option A CAnny
	int lowThreshold = 32;
	int ratio = 3;
	int kernel_size = 3;
	cv::Mat edges;
	cv::Canny(  imgGauss, edges, lowThreshold, lowThreshold*ratio, kernel_size );
	//ShowAndSave("edges", cols, rows, edges, baseDir,0);


//threshold canny
	//threshold Canny
	/*
	int lowThreshold = 30;
	int ratio = 3;
	int kernel_size = 3;
	*/
	if (false){
		window_name = "Threshold Canny";
		src_gray=imgGauss;

		  /// Create a window to display results
		cv::namedWindow( window_name, cv::WINDOW_NORMAL );
		cv::resizeWindow(window_name, cols / 2, rows / 4);

		int const max_value_lowThreshold = 255;

		  /// Create Trackbar to choose type of Threshold
		std::string trackbar_lowThreshold = "lowThreshold";
		  cv::createTrackbar( trackbar_lowThreshold,
		                  window_name, &lowThresholdG,
						  max_value_lowThreshold, Threshold_Canny);


		  std::string trackbar_ratio = "ratio";
		  int const max_value_ratio = 255;
		  cv::createTrackbar( trackbar_ratio,
		                  window_name, &ratioG,
		                  max_value_ratio, Threshold_Canny );


		  std::string trackbar_kernel_size = "kernel_size";
		  int const max_value_kernel_size = 7;
		  cv::createTrackbar( trackbar_kernel_size,
		                  window_name, &kernel_sizeG,
						  max_value_kernel_size, Threshold_Canny );

		  /// Call the function to initialize
		  Threshold_Canny( 0, 0 );

		  /// Wait until user finishes program
		  while(true)
		  {
		    int c;
		    c = cv::waitKey( 20 );
		    if( (char)c == 27 )
		      { break; }
		   }


	}


cv::Mat cdst;
	/*  Line detection
	 * white line on a black/gray background
	 * horizontal
	 * a minimum of vMinLine should be continuous and exposed,
	 * the line is extended to the limits of the image for evaluation porporsues
	 * start and end points of the line are found (extended line till there is no white )
	 */


	  std::vector<cv::Vec4i> line;
	  cv::HoughLinesP(edges, line, 1, CV_PI/180, 50, 50, 50 );
	  cv::Mat copy_edges=edges.clone();
	  for( size_t i = 0; i < line.size(); i++ )
	  {
	    cv::Vec4i l = line[i];
	    cv::line( copy_edges, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,255,255), 10, CV_AA);
	  }
	  ShowAndSave("detected lines", cols, rows, copy_edges, baseDir,0);



	  /*

void Threshold_HoughLinesP( int, void*)
	   */


		if (false){
			window_name = "Threshold Hough";
			src_gray=edges;

			  /// Create a window to display results
			cv::namedWindow( window_name, cv::WINDOW_NORMAL );
			cv::resizeWindow(window_name, cols / 2, rows / 4);

			int const max_value_rho = 255;

			  /// Create Trackbar to choose type of Threshold
			std::string trackbar_rho = "rho/10";
			  cv::createTrackbar( trackbar_rho,
			                  window_name, &rho,
							  max_value_rho, Threshold_HoughLinesP);


			  std::string trackbar_theta = "thetagrad";
			  int const max_value_theta = 360;
			  cv::createTrackbar( trackbar_theta,
			                  window_name, &theta,
			                  max_value_theta, Threshold_HoughLinesP );


			  std::string trackbar_threshold = "threshold";
			  int const max_threshold = 10000;
			  cv::createTrackbar( trackbar_threshold,
			                  window_name, &threshold,
							  max_threshold, Threshold_HoughLinesP );

			  std::string trackbar_minLineLength = "minLineLength";
			  int const max_minLineLength = cols;
			  cv::createTrackbar( trackbar_minLineLength,
			                  window_name, &minLineLength,
							  max_minLineLength, Threshold_HoughLinesP );

			  std::string trackbar_maxLineGap = "maxLineGap";
			  int const max_maxLineGap = cols;
			  cv::createTrackbar( trackbar_maxLineGap,
			                  window_name, &maxLineGap,
							  max_maxLineGap, Threshold_HoughLinesP );

			  /// Call the function to initialize
			  Threshold_HoughLinesP( 0, 0 );

			  /// Wait until user finishes program
			  while(true)
			  {
			    int c;
			    c = cv::waitKey( 20 );
			    if( (char)c == 27 )
			      { break; }
			   }


		}


		
		//get the angle of the lines
		std::vector<cv::Vec4i> straighLines;
		getAngle(line, straighLines);

		//longest horizontal line



	//5 fine contours

	  std::vector<std::vector<cv::Point> > contours;
	  std::vector<cv::Vec4i> hierarchy;
	  cv::findContours( edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	  // PLOT: countours
	  cv::Mat drawing = cv::Mat::zeros( edges.size(), CV_8UC3 );
	  cv::RNG rng(12345);
	  for( std::size_t i = 0; i< contours.size(); i++ )
	       {
	         cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	         drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
	       }

	  if (false){
		ShowAndSave("Contours", cols, rows, drawing, baseDir,1);

		}







	//anatomy + intensity+ lines
	// 5 Remove horizontal objects and small objects
		// 5.1) Fits an ellipse around a set of 2D points.
		/*
		const int minSize=5;
		const int minAngle=10;
		std::vector<cv::RotatedRect> rotatedRect(contours.size());
		for( std::vector<std::vector<cv::Point>>::iterator it=contours.begin(); it!=contours.end());{
                //std::size_t i = 0 ; i<contours.size() ; i++ )

			if( contours[it].size() > minSize ){
				rotatedRect[it] = fitEllipse( cv::Mat(contours[it]) );


				if (abs(rotatedRect[it].angle)<minAngle){
					it++;
				}
				else{
					rotatedRect.erase(rotatedRect.begin()+i); //TODO check .beging
					contours.erase(contours.begin()+i); //TODO check .beging
				}
			}
		}

		//PLOT: Draw contours + rotated rects + ellipses
		  cv::Mat noHoriz = cv::Mat::zeros( imgGauss.size(), CV_8UC3 );
		  for( size_t i = 0; i< contours.size(); i++ )
		     {
		       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		       // contour
		       drawContours( noHoriz, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
		       // ellipse
		       ellipse( noHoriz, rotatedRect[i], color, 2, 8 );
		       // rotated rectangle
		       cv::Point2f rect_points[4]; rotatedRect[i].points( rect_points );
		       for( int j = 0; j < 4; j++ )
		          line( noHoriz, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
		     }

		  /// Show in a window

			windowName = "Contours";
			cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
			cv::resizeWindow(windowName, cols/2, rows/4);
			imshow(windowName, noHoriz);
			cv::waitKey(0);


*/


	// Fitting in a line https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html




	/*
	 * 	/// 1.b. Creating circles
	 * 		cv::Mat image = cv::Mat::zeros( rows, columns, CV_8UC3 );
	int thickness = -1;
	int lineType = 8;
	cv::circle( image,
			cv::Point(  columns/2.0,rows/2.0),
			10.0,
			cv::Scalar( 255, 0, 255 ),
			thickness,
			lineType );
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", image );

	cv::waitKey(0);

	 */


	  std::string msg= "[   {      \"Box\": [         1286,         802,         2054,         809,        1362,         2545,         2033,         2566      ],      \"Distance To line\": 0,      \"Error Angle\": -10,      \"Next Car\": [         {            \"Distance\": 200,            \"Closer points\": [               2033,               2335,               2241,               2352            ]         }      ]   },   {      \"Box\": [         2497,         788,         3154,         809,         2222,         2490,         2863,         2635      ],      \"Distance To line\": 10,      \"Error Angle\": 20,      \"Next Car\": [         {            \"Distance\": 200,            \"Closer points\": [               2033,               2335,               2241,               2352            ]         },         {            \"Distance\": 50,            \"Closer points\": [               3189,               1003,               3258,               1003            ]         }      ]   }]";
	  std::ofstream outputJsonResult;
	  std::string result = baseDir + "result.json";
	  outputJsonResult.open(&result[0], std::ofstream::out | std::ofstream::trunc);
	  outputJsonResult<< msg;
	  outputJsonResult.close();

	cv::destroyAllWindows(); //destroy the created window


	std::cout << "+++END Drone_Parking_Evaluation 2!!!" << std::endl;
	return 0;



}



