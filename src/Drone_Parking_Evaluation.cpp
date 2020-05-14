//============================================================================
// Name        : Drone_Parking_Evaluation.cpp
// Author      : ivan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
//using namespace cv;
//using namespace std;
int main() {
	std::cout << "!!!start Drone_Parking_Evaluation!!!" << std::endl;

	const std::string rawImgAdd="data/Situation_3_Auswahl/Situation_3_nah.JPG";
	const std::string baseDir="output/";
	cv::Mat raw;
	raw=cv::imread( rawImgAdd  );
	if (raw.empty())
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	const int cols= raw.cols;
	const int rows= raw.rows;
	std::string windowName;
	//original image

	 windowName = "Raw";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/2);
	imshow(windowName, raw);
	cv::waitKey(0);

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

	//PLOT: gray image

	windowName = "imgGray";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/4);
	imshow(windowName, imgGray);
	imwrite( baseDir+windowName+".png", imgGray );
	cv::waitKey(0);

	//3 Gausian filter
	cv::Mat imgGauss;
	GaussianBlur( imgGray, imgGauss, cv::Size( 15, 15 ), 3, 0 ); //TODO: try size 5 sugested by open cv

	//PLOT: gray image
	windowName = "imgGauss";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/4);
	imshow(windowName, imgGauss);
	imwrite( baseDir+windowName+".png", imgGauss );
	cv::waitKey(0);

	//4 Edge detection
	//option A CAnny
	int lowThreshold = 30;
	const int ratio = 3;
	const int kernel_size = 3;
	cv::Mat edges;
	cv::Canny(  imgGauss, edges, lowThreshold, lowThreshold*ratio, kernel_size );

	//PLOT: edges

	windowName = "edges";
	cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
	cv::resizeWindow(windowName, cols/2, rows/4);
	imshow(windowName, edges);
	imwrite( baseDir+windowName+".png", edges );
	cv::waitKey(0);


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

		windowName = "Contours";
		cv::namedWindow(windowName, cv::WINDOW_NORMAL  );
		cv::resizeWindow(windowName, cols/2, rows/4);
		imshow(windowName, drawing);
		imwrite( baseDir+windowName+".png", drawing );
		cv::waitKey(0);



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
	cv::destroyAllWindows(); //destroy the created window



	return 0;



}
