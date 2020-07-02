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
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

// Global variables
std::vector<std::vector<cv::Point>> cont;
cv::Point2f line_points[2];
cv::Mat imgGauss;
std::string window_name;
int threshold_value = 0;
int threshold_type = 3;
bool tTune=0;
std::string testImag="mittel"; //fern //nah

int const max_BINARY_value = 255;

cv::Mat src, src_gray, dst;
int cols;
int rows;
bool tCars=0;
bool tLines=0;

void ShowAndSave(const std::string& windowName, const int cols, const int rows,
		const cv::Mat& Img, const std::string& baseDir, bool wait) {
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::resizeWindow(windowName, cols / 2, rows / 4);
	imshow(windowName, Img);
	imwrite(baseDir + windowName + ".png", Img);
	if (wait)
		cv::waitKey(0);
}

//threshold Canny
int lowThresholdG = 18;//25;
int ratioG = 3;
int kernel_sizeG = 2;
int erosion_elem = 0;
int erosion_size = 18;
int dilation_elem = 0;
int dilation_size = 10;//7;//15;
int const max_elem = 2;
int const max_kernel_size = 21;
int morph_operator = 0;
int morph_size = 0;
int iterations_openin=8;
int threshold_value_shadow = 100;
int threshold_type_shadow = 0;
int const max_BINARY_value_shadow = 255;
int  maxArea = 800000;
int minArea = 60000;//100000;
int action = 100000; //to run the thresholds
int stepView = 10;

//solutions
std::vector<std::vector<cv::Point> > contours;
std::vector<cv::Vec4i> hierarchy;


void DrawAllContours(const std::vector<std::vector<cv::Point> > & contours2,
		cv::Mat& drawing) {

	cv::RNG rng(12345);
	for (std::size_t i = 0; i < contours2.size(); i++) {
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		drawContours(drawing, contours2, i, color, 10, 8, hierarchy, 0, cv::Point());
	}
}

void DrawAllContours(const std::vector<cv::Point> & contours2,
		cv::Mat& drawing) {
	std::vector<std::vector<cv::Point>>  contours3;
	contours3.push_back(contours2);
	DrawAllContours(contours3,drawing);
}

void DrawRotatedRect(const std::vector<std::vector<cv::Point> >& contours2,
		cv::Mat& drawing,std::vector<cv::RotatedRect>& minRect) {
	cv::RNG rng(12345);
	//std::vector<cv::RotatedRect> minRect(contours2.size());
	for (std::size_t i = 0; i < contours2.size(); i++) {
		minRect.push_back( cv::minAreaRect(cv::Mat(contours2[i])));
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		// rotated rectangle
		cv::Point2f rect_points[4];
		minRect.at(i).points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 20,
					8);
	}
}

void DrawRotatedRect(const std::vector<std::vector<cv::Point> >& contours2,
		cv::Mat& drawing) {
	std::vector<cv::RotatedRect> minRect;
	DrawRotatedRect(contours2,drawing,minRect);
}


void voidMethod( int, void*)
{
}

void getScreenResolution(int &width, int &height) {
	width= 1376;
	height=780;

}


void split1(const std::string& str, std::vector<std::vector<std::string>>& vecstring)
{
	std::vector<std::string> temp;
	std::size_t Ocoma=0,Ncoma=0;

	Ncoma =str.find(",");

	while (Ncoma!=std::string::npos)
	{
		temp.push_back(str.substr(Ocoma, Ncoma-Ocoma));
		Ocoma=Ncoma+1;
		Ncoma =str.find(",",Ocoma);
	}
	temp.push_back(str.substr(Ocoma, str.size()-1));
	vecstring.push_back(temp);
}

void Threshold_Canny(const std::string & rawImgAdd, const std::string & mWindow_name, const cv::Point & location, const int & windowWidth, const int &windowHeight)
{
	std::cout << "Data: "<< rawImgAdd<<".JPG / .CSV" << std::endl;

	// File pointer
	std::fstream fin;

	// Open an existing file
	try{
		fin.open(rawImgAdd+".csv", std::ios::in);
	}catch(const std::exception& e)
	{
		std::cout<<"ERROR: no valid directory for the .CSV file "<<rawImgAdd<<".csv"<<std::endl;
		abort();
	}

	// Read the Data from the file
	// as String Vector
	std::string line, temp;
	std::vector<std::vector<std::string>> words;
	while (fin >> temp) {

		getline(fin, line);
		split1(line, words);
	}
	//std::cout<<words.at(71).at(1)<<"\n";


	cv::Mat raw=cv::imread( rawImgAdd +".JPG" );
	if (raw.empty())
	{
		std::cout << "ERROR: Could not open or find the image:  "<<rawImgAdd<<".JPG" << std::endl;
		abort();
	}
	cols= raw.cols;
	rows= raw.rows;

	//2 RGB2Gray
	cv::Mat imgGray;
	cvtColor(raw, imgGray, cv::COLOR_BGR2GRAY);

	//3 Gausian filter
	GaussianBlur( imgGray, imgGauss, cv::Size( 31, 31 ), 3, 0 ); //TODO: try size 5 sugested by open cv

	if (kernel_sizeG<3)
		kernel_sizeG=3;
	cv::Mat dst_Canny;
	kernel_sizeG= 2 * ( (int)(kernel_sizeG / 2.0f) ) + 1;
	cv::Canny(  imgGauss, dst_Canny, lowThresholdG, lowThresholdG*ratioG, kernel_sizeG );

	//dilate
	dst=dst_Canny.clone();
	cv::Mat src_dilate=dst_Canny;//.clone();
	int dilation_type;
	if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
	else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
	else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

	cv::Mat element = getStructuringElement( dilation_type,
			cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
			cv::Point( dilation_size, dilation_size ) );
	/// Apply the dilation operation
	cv::Mat dst_dilation;
	cv::dilate( src_dilate, dst_dilation, element );


	//Flood fill from pixel (0, 0).
	cv::Mat im_floodfill = dst_dilation.clone();
	cv::floodFill(im_floodfill, cv::Point(0,0), cv::Scalar(255));
	//Flood fill from pixel (cols/2,0).
	cv::floodFill(im_floodfill, cv::Point(cols/2,0), cv::Scalar(255));
	cv::floodFill(im_floodfill, cv::Point(cols-1,0), cv::Scalar(255));
	cv::floodFill(im_floodfill, cv::Point(cols-1,rows-1), cv::Scalar(255));
	cv::floodFill(im_floodfill, cv::Point(0,rows-1), cv::Scalar(255));

	//Invert the flood filled image
	cv::Mat im_floodfill_inv;
	cv::bitwise_not(im_floodfill, im_floodfill_inv);


	//Combine the thresholded image with the inverted flood filled image using bitwise OR operation to obtain the final foreground mask with holes filled in.
	cv::Mat im_Combine = (dst_dilation | im_floodfill_inv);

	//erode
	//cv::Mat src_erosion=dst_dilation.clone();
	cv::Mat src_erosion=im_Combine;//.clone();
	int erosion_type;
	if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
	else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
	else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

	element = getStructuringElement( erosion_type,
			cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
			cv::Point( erosion_size, erosion_size ) );

	/// Apply the erosion operation
	cv::Mat dst_erosion;
	cv::erode( src_erosion, dst_erosion, element );
	//imshow( "Erosion Demo", erosion_dst );


	//Remove Lines using opening with vertical lines

	// Since MORPH_X : 2,3,4,5 and 6
	int operation = morph_operator + 2;
	//vertical rectangle element
	/*
   cv::Mat_ <float> kernel_element = cv::Mat::zeros( 5, 5, CV_8U);
   kernel_element << 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0,0,0,1,0,0,	0,0,1,0,0;
	 */
	int kdata[] = {0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0,0,0,1,0,0,	0,0,1,0,0};
	cv::Mat kernel(5,5,CV_8U, kdata);
	/*
   float kernel_element_data[] = {0, 1, 0, 0, 1, 0, 0, 1, 0};
   cv::Mat kernel_element(3,3,CV_32F, kernel_element_data);
	 */
	cv::Mat element_open = getStructuringElement( cv::MORPH_ELLIPSE,
			cv::Size( 2*10 + 1, 2*10+1 ),
			cv::Point( 10, 10 ) );

	//opening to remove the line of connecting cars
	cv::Mat dst_opening;


	cv::Mat src_opening=dst_erosion;//.clone();
	cv::morphologyEx(src_opening, dst_opening, operation, element_open,cv::Point(-1,-1), iterations_openin); //TODO:borders?


	//get gray image from the selected areas

	//multyply one to one with the gray image
	cv::Mat src_bitwise_and=dst_opening.clone();
	cv::Mat dst_bitwise_and;
	cv::bitwise_and(src_bitwise_and, imgGauss, dst_bitwise_and);

	//threshold on the image to remove the shadows,
	cv::Mat src_Thresnold_shadow=dst_bitwise_and;//.clone();
	cv::Mat dst_Thresnold_shadow;
	cv::threshold( src_Thresnold_shadow, dst_Thresnold_shadow, threshold_value_shadow, max_BINARY_value_shadow,threshold_type_shadow );

	//close objects

	//opening to remove the line of connecting cars
	cv::Mat dst_closing;
	cv::Mat src_opening2=dst_Thresnold_shadow;//.clone();
	cv::morphologyEx(src_opening2, dst_closing, 3, element_open,cv::Point(-1,-1), iterations_openin); //TODO:borders?

	//Flood fill from pixel (0, 0).
	cv::Mat im_floodfill_shadow = src_opening2.clone();
	cv::floodFill(im_floodfill_shadow, cv::Point(0,0), cv::Scalar(255));


	//Invert the flood filled image
	cv::Mat im_floodfill_inv_shadow;
	cv::bitwise_not(im_floodfill_shadow, im_floodfill_inv_shadow);


	//Combine the thresholded image with the inverted flood filled image using bitwise OR operation to obtain the final foreground mask with holes filled in.

	//cv::Mat im_Combine_shadow = (dst_Thresnold_shadow | im_floodfill_inv_shadow);
	//cv::Mat im_Combine_shadow = (dst_Thresnold_shadow | src_opening2);
	cv::Mat im_Combine_shadow =src_opening2.clone();


	//5 find contours

	cv::Mat src_contours=  im_Combine_shadow.clone();
	std::vector<std::vector<cv::Point>> contours2;


	cv::findContours( src_contours, contours2, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	//filter contourns using size

	for (auto it = contours2.begin(); it!=contours2.end(); it++ )
	{

		double area= contourArea(*it);
		if (area>maxArea || area <minArea)

		{contours2.erase(it);  //TODO find abetter solution
		it--;
		}

	}
	contours=contours2; //needed global var

	cv::Mat drawing = cv::Mat::zeros( dst.size(), CV_8UC3 );
	DrawAllContours(contours2, drawing);

	DrawRotatedRect(contours2, drawing);
	cv::Mat dst_end=drawing.clone();

	if (tTune){
		// draw in the raw image

		cv::Mat drawingOnRaw = raw.clone();
		DrawAllContours(contours, drawingOnRaw);
		DrawRotatedRect(contours, drawingOnRaw);


		/// Create a window to display results
		cv::namedWindow( mWindow_name, cv::WINDOW_NORMAL );
		cv::resizeWindow(mWindow_name, windowWidth, windowHeight);
		cv::moveWindow	(mWindow_name, location.x,location.y);

		//using stepView to choose the step to view

		cv::Mat stepViewImage;
		switch (stepView) {
		case 0:
			stepViewImage=dst_Canny;
			break;
		case 1:
			stepViewImage=dst_dilation;
			break;
		case 2:
			stepViewImage=im_floodfill;
			break;
		case 3:
			stepViewImage=im_Combine;
			break;
		case 4:
			stepViewImage=dst_erosion;
			break;
		case 5:
			stepViewImage=dst_bitwise_and;
			break;
		case 6:
			stepViewImage=dst_Thresnold_shadow;
			break;
		case 7:
			stepViewImage=dst_opening;
			break;
		case 8:
			stepViewImage=src_opening2;
			break;
		case 9:
			stepViewImage=im_Combine_shadow;
			break;



		default:
			stepViewImage=drawingOnRaw;
			break;
		}

		cv::imshow( mWindow_name, stepViewImage );

	}
}

void CannyLunch( int, void*)
{
	int width,height;
	getScreenResolution(width, height);
	int windowHeight=height/3;
	int windowWidth=windowHeight*1.5;

	cv::Point location1=cv::Point(2*width/3,0);
	cv::Point location2=cv::Point(2*width/3,height/3);
	cv::Point location3=cv::Point(2*width/3,height*2/3);
	cv::Point location4=cv::Point(width/3+100,0);
	cv::Point location5=cv::Point(width/3+100,height*2/3);

	if (testImag=="fern"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_fern";
		Threshold_Canny(rawImgAdd,"cannyLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_fern";
		Threshold_Canny(rawImgAdd2,"cannyLunch2",location2,windowWidth,windowHeight);
		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_fern";
		Threshold_Canny(rawImgAdd3,"cannyLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_fern";
		Threshold_Canny(rawImgAdd4,"cannyLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_fern";
		Threshold_Canny(rawImgAdd5,"cannyLunch5",location5,windowWidth,windowHeight);
	}

	if (testImag=="mittel"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_mittel";
		Threshold_Canny(rawImgAdd,"cannyLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_mittel";
		Threshold_Canny(rawImgAdd2,"cannyLunch2",location2,windowWidth,windowHeight);

		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_mittel";
		Threshold_Canny(rawImgAdd3,"cannyLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_mittel";
		Threshold_Canny(rawImgAdd4,"cannyLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_mittel";
		Threshold_Canny(rawImgAdd5,"cannyLunch5",location5,windowWidth,windowHeight);
	}
	if (testImag=="nah"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_nah";
		Threshold_Canny(rawImgAdd,"cannyLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_nah";
		Threshold_Canny(rawImgAdd2,"cannyLunch2",location2,windowWidth,windowHeight);

		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_nah";
		Threshold_Canny(rawImgAdd3,"cannyLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_nah";
		Threshold_Canny(rawImgAdd4,"cannyLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_nah";
		Threshold_Canny(rawImgAdd5,"cannyLunch5",location5,windowWidth,windowHeight);
	}
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
		//float angle = atan2(p1.y - p2.y, p1.x - p2.x) * 180 / CV_PI;

		float angle;
		if (p1.x<p2.x)
			angle = atan2((p2.y - p1.y), (p2.x - p1.x)) * 180 / CV_PI;
		else
			angle = atan2((p1.y - p2.y), (p1.x - p2.x)) * 180 / CV_PI;

		angles.push_back(angle);
		if (angle < maxAngle && angle > -maxAngle)
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


struct AreaContourns
{
	double area;
	std::vector<cv::Point>contourn;

	AreaContourns(double a, const std::vector<cv::Point> & c) : area(a), contourn(c) {}

	bool operator < (const AreaContourns& ac) const
	{
		return (area > ac.area);
	}
};

//threshold HoughLinesP
int rho=10;
int theta= 1;
int threshold=50;//10000;
int minLineLength=50;//3000;
int maxLineGap=50;//700;

void plotLines(const std::vector<cv::Vec4i>& lines, cv::Mat copy_src, cv::Scalar color, bool plot=0 ) {
	for (size_t i = 0; i < lines.size(); i++) {
		cv::Vec4i l = lines[i];
		cv::line(copy_src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
				color, 10, CV_AA);
	}
	if(plot)
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

void DrawLine(cv::Point2f* line_points, cv::Mat& draw_countours) {
	//print the center line of the rotated box
	cv::RNG rng(12345);
	cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
			rng.uniform(0, 255));
	line(draw_countours, line_points[0], line_points[1], color, 50, 8);
}

void DrawLine(std::vector<std::vector<cv::Point2f>>vector_line_points, cv::Mat& draw_countours) {
	for(auto singleVector:vector_line_points){
		cv::Point2f singleline[2] ={singleVector.at(0),singleVector.at(1)};
		DrawLine(singleline, draw_countours);
	}
}

float euclideanDist(cv::Point2f& p, cv::Point2f& q) {
	cv::Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

void getCentralLine(const cv::RotatedRect& lineRotatedRect, cv::Point2f* central_line_points) {
	cv::Point2f rect_points[4];
	lineRotatedRect.points(rect_points);
	//orginize the points to follow 	2|1
	//3|4
	//find cuad2	the smaller
	cv::Point2f cuad1 = rect_points[0];
	cv::Point2f cuad4 = rect_points[1];
	cv::Point2f cuad2 = rect_points[2];
	cv::Point2f cuad3 = rect_points[3];
	//Right group
	if (rect_points[2].x > cuad1.x) {
		cuad1 = rect_points[2];
		cuad2 = rect_points[0];
	}
	if (rect_points[3].x > cuad4.x) {
		cuad4 = rect_points[3];
		cuad3 = rect_points[1];
	}
	//Right top bottom
	if (cuad1.y > cuad4.y) {
		cv::Point2f temp = cuad1;
		cuad1 = cuad4;
		cuad4 = temp;
	}
	// left Group top bottom
	if (cuad2.y > cuad3.y) {
		cv::Point2f temp = cuad2;
		cuad2 = cuad3;
		cuad3 = temp;
	}
	if( euclideanDist(cuad1,cuad2)< euclideanDist(cuad2,cuad3))
	{
		central_line_points[0].x = cuad2.x + (cuad1 - cuad2).x / 2;
		central_line_points[0].y = cuad2.y + (cuad1 - cuad2).y / 2;
		central_line_points[1].x = cuad3.x + (cuad4 - cuad3).x / 2;
		central_line_points[1].y = cuad3.y + (cuad4 - cuad3).y / 2;

	}else
	{
		central_line_points[0].x = cuad3.x + (cuad2 - cuad3).x / 2;
		central_line_points[0].y = cuad3.y + (cuad2 - cuad3).y / 2;
		central_line_points[1].x = cuad4.x + (cuad1 - cuad4).x / 2;
		central_line_points[1].y = cuad4.y + (cuad1 - cuad4).y / 2;
	}


}

void getCentralLine(const std::vector<cv::Point>& lineRotatedRect, cv::Point2f* central_line_points) {

	cv::RotatedRect minRect;
	minRect = cv::minAreaRect(lineRotatedRect);
	getCentralLine(minRect, central_line_points);

}

void getLineEqu(cv::Point& p1,cv::Point& p2,double &inc,float & k){

	if (p1.x<p2.x)
	{double diven=(p2.x - p1.x);
	double divsor=(p2.y - p1.y);
	inc = divsor/diven;
	}
	else
	{inc = (p1.y - p2.y)/ (p1.x - p2.x);
	}
	k =p1.y-(p1.x)*inc;

}

void getLineEqu(cv::Point2f& p1,cv::Point2f& p2,double &inc,float & k){

	if (p1.x<p2.x)
	{double diven=(p2.x - p1.x);
	double divsor=(p2.y - p1.y);
	inc = divsor/diven;
	}
	else
	{inc = (p1.y - p2.y)/ (p1.x - p2.x);
	}
	k =p1.y-(p1.x)*inc;

}

void Threshold_HoughLinesP(const std::string & rawImgAdd, const std::string & mWindow_name, const cv::Point & location, const int & windowWidth, const int &windowHeight, cv::RotatedRect &lineRotatedRect)
{
	//std::cout << "Threshold_HoughLinesP Data: "<< rawImgAdd << std::endl;

	// File pointer
	std::fstream fin;

	// Open an existing file
	fin.open(rawImgAdd+".csv", std::ios::in);

	// Read the Data from the file
	// as String Vector
	std::string line, temp;
	std::vector<std::vector<std::string>> words;
	while (fin >> temp) {



		getline(fin, line);
		split1(line, words);
	}
	//std::cout<<words.at(71).at(1)<<"\n";


	cv::Mat raw=cv::imread( rawImgAdd +".JPG" );
	if (raw.empty())
	{
		std::cout << "ERROR: Could not open or find the image:  "<<rawImgAdd<<".JPG" << std::endl;
		abort();
	}
	cols= raw.cols;
	rows= raw.rows;

	//2 RGB2Gray
	cv::Mat imgGray;
	cvtColor(raw, imgGray, cv::COLOR_BGR2GRAY);
	//ShowAndSave("imgGray", cols, rows, imgGray, baseDir,0);

	//3 Gausian filter
	GaussianBlur( imgGray, imgGauss, cv::Size( 31, 31 ), 3, 0 ); //TODO: try size 5 sugested by open cv
	//ShowAndSave("imgGauss", cols, rows, imgGauss, baseDir,0);


	if (kernel_sizeG<3)
		kernel_sizeG=3;
	cv::Mat dst_Canny;
	kernel_sizeG= 2 * ( (int)(kernel_sizeG / 2.0f) ) + 1;
	cv::Canny(  imgGauss, dst_Canny, lowThresholdG, lowThresholdG*ratioG, kernel_sizeG );

	//dilate
	dst=dst_Canny.clone();

	double thetarad= theta/10*CV_PI/180;
	thetarad=CV_PI/180;
	double drho =rho/10;
	drho=1;


	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(dst_Canny, lines, drho, thetarad, threshold, minLineLength, maxLineGap );

	cv::Mat copy_src=imgGray.clone();

	plotLines(lines, copy_src, cv::Scalar(255, 255, 255));
	std::vector<cv::Vec4i> straighLines;
	getAngle(lines, straighLines, 5);
	plotLines(straighLines, copy_src, cv::Scalar(100, 100, 255));

	// get longer lines
	size_t numlines =5;

	std::vector<DistanceLines> longerLines;
	std::vector<DistanceLines> linesWLenghts;

	for (cv::Vec4i line : straighLines)
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

	std::vector<cv::Vec4i> endToEndLines;
	//getEndtoEndLines(longerLines,endToEndLines, row);
	for(auto longerLine: longerLines){
		//getInclAndK
		cv::Point p1, p2;
		p1 = cv::Point(longerLine.line[0], longerLine.line[1]);
		p2 = cv::Point(longerLine.line[2], longerLine.line[3]);
		double inc=0;
		float k=0;
		getLineEqu(p1,p2, inc,k);

		//endToEndLines
		float px0=0, py0,pxend=cols,pyend;
		py0=k;
		pyend=pxend*inc+k;
		endToEndLines.push_back(cv::Vec4i {px0,py0,pxend,pyend});

	}

	plotLines(endToEndLines,copy_src,cv::Scalar(50, 50, 255));
	plotLines(longerLines, copy_src, cv::Scalar(100, 200, 255));

	//canvas for all straig lines and end to end of the longer lines
	cv::Mat emptyMat= cv::Mat::zeros(rows,cols,CV_8UC1); //TODO:TIPE?
	cv::Mat plotendToEndLines=emptyMat.clone();
	cv::Mat plotstraighLines=emptyMat.clone();


	plotLines(straighLines, plotstraighLines, cv::Scalar(100, 200, 255),0);
	plotLines(endToEndLines, plotendToEndLines, cv::Scalar(100, 200, 255),0);

	//bitwise_and
	cv::Mat dst_bitwise_straight_endtoend;
	cv::bitwise_and(plotstraighLines, plotendToEndLines, dst_bitwise_straight_endtoend);

	// close

	int dil_ellip=15;
	cv::Mat element_open2 = getStructuringElement( 0,
			cv::Size( 2*dil_ellip + 1, 2*dil_ellip+1 ),
			cv::Point( dil_ellip, dil_ellip ) );

	//opening to remove the line of connecting cars
	int  iterations_openin_line=1;
	cv::Mat dst_opening;
	cv::Mat src_opening=dst_bitwise_straight_endtoend.clone();
	/// Apply the dilation operation
	cv::Mat dst_dilation2;
	cv::dilate( src_opening, dst_dilation2, element_open2 ); //TODO: takes to much time Due to the Element shape

	//get contours
	cv::Mat dst_contours2;
	cv::Mat src_contours2=dst_dilation2.clone();
	std::vector<std::vector<cv::Point> > contours2;
	std::vector<cv::Vec4i> hierarchy2;

	cv::findContours( src_contours2, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

	//get the bigger contour
	std::vector<AreaContourns> lineAreaContourn;

	for (std::vector<cv::Point> contour : contours2)
	{
		lineAreaContourn.push_back( AreaContourns(cv::contourArea(contour),contour));
	}

	std::sort(lineAreaContourn.begin(), lineAreaContourn.end());

	//fit a line on the contour of the biggest contour
	// Finds a rotated rectangle of the minimum area enclosing the input 2D point set
	lineRotatedRect = cv::minAreaRect(lineAreaContourn.at(0).contourn);

	getCentralLine(lineRotatedRect,line_points);
	// Middle points
	hierarchy=hierarchy2;

	//print the bigger contour
	cv::Mat draw_countours = cv::Mat::zeros( dst.size(), CV_8UC3 );
	DrawAllContours(lineAreaContourn.at(0).contourn, draw_countours);

	//print the bigger contour rotated Box

	cont.push_back(lineAreaContourn.at(0).contourn);
	DrawRotatedRect(cont, draw_countours);

	//print the center line of the rotated box
	DrawLine(line_points, draw_countours);
	//cv::imshow(window_name,draw_countours);

	if (tTune){
		// draw in the raw image

		cv::Mat drawingOnRaw = raw.clone();
		DrawRotatedRect(cont, drawingOnRaw);
		DrawLine(line_points, drawingOnRaw);


		/// Create a window to display results
		cv::namedWindow( mWindow_name, cv::WINDOW_NORMAL );
		cv::resizeWindow(mWindow_name, windowWidth, windowHeight);
		cv::moveWindow	(mWindow_name, location.x,location.y);

		//using stepView to choose the step to view

		cv::Mat stepViewImage;
		switch (stepView) {
		/*case 0:
			stepViewImage=dst_Canny;
			break;
		case 1:
			stepViewImage=dst_dilation;
			break;
		case 2:
			stepViewImage=im_floodfill;
			break;
		case 3:
			stepViewImage=im_Combine;
			break;
		case 4:
			stepViewImage=dst_erosion;
			break;
		case 5:
			stepViewImage=dst_bitwise_and;
			break;
		case 6:
			stepViewImage=dst_Thresnold_shadow;
			break;
		case 7:
			stepViewImage=dst_opening;
			break;
		case 8:
			stepViewImage=src_opening2;
			break;
		 */		case 9:
			 stepViewImage=draw_countours;
			 break;



		 default:
			 stepViewImage=drawingOnRaw;
			 break;
		}

		cv::imshow( mWindow_name, stepViewImage );

	}

}
void Threshold_HoughLinesP(const std::string & rawImgAdd, const std::string & mWindow_name, const cv::Point & location, const int & windowWidth, const int &windowHeight )
{
	cv::RotatedRect trotRec;
	Threshold_HoughLinesP(rawImgAdd, mWindow_name, location,  windowWidth, windowHeight, trotRec);

}


void houghLunch( int, void*)
{
	int width,height;
	getScreenResolution(width, height);
	int windowHeight=height/3;
	int windowWidth=windowHeight*1.5;

	cv::Point location1=cv::Point(2*width/3,0);
	cv::Point location2=cv::Point(2*width/3,height/3);
	cv::Point location3=cv::Point(2*width/3,height*2/3);
	cv::Point location4=cv::Point(width/3+100,0);
	cv::Point location5=cv::Point(width/3+100,height*2/3);

	if (testImag=="fern"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_fern";
		Threshold_HoughLinesP(rawImgAdd,"houghLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_fern";
		Threshold_HoughLinesP(rawImgAdd2,"houghLunch2",location2,windowWidth,windowHeight);
		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_fern";
		Threshold_HoughLinesP(rawImgAdd3,"houghLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_fern";
		Threshold_HoughLinesP(rawImgAdd4,"houghLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_fern";
		Threshold_HoughLinesP(rawImgAdd5,"houghLunch5",location5,windowWidth,windowHeight);
	}

	if (testImag=="mittel"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_mittel";
		Threshold_HoughLinesP(rawImgAdd,"houghLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_mittel";
		Threshold_HoughLinesP(rawImgAdd2,"houghLunch2",location2,windowWidth,windowHeight);

		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_mittel";
		Threshold_HoughLinesP(rawImgAdd3,"houghLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_mittel";
		Threshold_HoughLinesP(rawImgAdd4,"houghLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_mittel";
		Threshold_HoughLinesP(rawImgAdd5,"houghLunch5",location5,windowWidth,windowHeight);

	}
	if (testImag=="nah"){
		//first image
		std::string rawImgAdd="data/Situation_1_Auswahl/Situation_1_nah";
		Threshold_HoughLinesP(rawImgAdd,"houghLunch1",location1,windowWidth,windowHeight);

		//second image
		std::string rawImgAdd2="data/Situation_2_Auswahl/situation_2_nah";
		Threshold_HoughLinesP(rawImgAdd2,"houghLunch2",location2,windowWidth,windowHeight);

		//3 image
		std::string rawImgAdd3="data/Situation_3_Auswahl/Situation_3_nah";
		Threshold_HoughLinesP(rawImgAdd3,"houghLunch3",location3,windowWidth,windowHeight);

		//4 image
		std::string rawImgAdd4="data/Situation_4_Auswahl/Situation_4_nah";
		Threshold_HoughLinesP(rawImgAdd4,"houghLunch4",location4,windowWidth,windowHeight);

		//5 image
		std::string rawImgAdd5="data/Situation_5_Auswahl/Situation_5_nah";
		Threshold_HoughLinesP(rawImgAdd5,"houghLunch5",location5,windowWidth,windowHeight);
	}


}




int main(int argc, char *argv[])
{
	std::cout << "!!!start Drone_Parking_Evaluation !!!" << std::endl;
	const std::string baseDir="output/";

	if (argc>1)
	{
		int com=strcmp(argv[1],"-tune");
		if (com==0)
			tTune=1;

		if (argc==3)
		{
			com=strcmp(argv[2],"nah");
			if (com==0)
				testImag="nah";

			com=strcmp(argv[2],"mittel");
			if (com==0)
				testImag="mittel";

			com=strcmp(argv[2],"fern");
			if (com==0)
				testImag="fern";
		}
	}
	if (tTune){



		/// Create a window to display results
		int width, height;
		getScreenResolution(width, height);
		std::string trackbar_window_name="Trackbar Canny";
		cv::namedWindow( trackbar_window_name, cv::WINDOW_NORMAL );
		cv::resizeWindow(trackbar_window_name, width/3, height);
		cv::moveWindow	(	trackbar_window_name,0,0) ;

		int const max_value_lowThreshold = 255;

		/// Create Trackbar to choose type of Threshold
		std::string trackbar_lowThreshold = "lowThreshold";
		cv::createTrackbar( trackbar_lowThreshold,
				trackbar_window_name, &lowThresholdG,
				max_value_lowThreshold, voidMethod);


		std::string trackbar_ratio = "ratio";
		int const max_value_ratio = 5;
		cv::createTrackbar( trackbar_ratio,
				trackbar_window_name, &ratioG,
				max_value_ratio, voidMethod );
		/*
		/// Create Dilation Trackbar
		cv::createTrackbar( "dilation_Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", trackbar_window_name,
				&dilation_elem, max_elem,
				voidMethod );
		 */
		cv::createTrackbar( "dilation_Kernel size:\n 2n +1", trackbar_window_name,
				&dilation_size, max_kernel_size,
				voidMethod );
		/*
		/// Create Erosion Trackbar
		cv::createTrackbar( "erosion_Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", trackbar_window_name,
				&erosion_elem, max_elem,
				voidMethod );
		 */
		cv::createTrackbar( "erosion_Kernel size:\n 2n +1", trackbar_window_name,
				&erosion_size, max_kernel_size,
				voidMethod );
		/*
		/// Create Trackbar to select Morphology operation
		int const max_operator = 4;
		cv::createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", trackbar_window_name,
				&morph_operator, max_operator, voidMethod );
		 */
		/// Create Trackbar to choose kernel size
		cv::createTrackbar( "iterations_openin", trackbar_window_name,
				&iterations_openin, max_kernel_size,
				voidMethod );
		/*
		/// Create Trackbar to choose type of Threshold
		std::string trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
		cv::createTrackbar( trackbar_type,
				trackbar_window_name, &threshold_type_shadow,
				4, voidMethod);
		 */
		std::string trackbar_value = "threshold_value_shadow";

		int const max_value = 255;
		cv::createTrackbar( trackbar_value,
				trackbar_window_name, &threshold_value_shadow,
				max_value, voidMethod );

		/// Create Trackbar to max area boxes
		const int  max_maxArea = 1000000;
		cv::createTrackbar("maxArea", trackbar_window_name,
				&maxArea, max_maxArea,
				voidMethod );

		/// Create Trackbar to min area boxes
		const int max_minArea = 1000000;
		cv::createTrackbar( "minArea", trackbar_window_name,
				&minArea, max_minArea,
				voidMethod );

		/// Create step view
		const int voidMaxStep=10;
		cv::createTrackbar( "step view", trackbar_window_name,
				&stepView, voidMaxStep,
				voidMethod );

		/// Create actions
		const int voidMax=9;
		cv::createTrackbar( "action", trackbar_window_name,
				&action, voidMax,
				CannyLunch );

		/// Call the function to initialize
		CannyLunch( 0, 0 );

		/// Wait until user finishes program
		while(true) //TODO Change for true if want to tunne
		{
			int c;
			c = cv::waitKey( 20 );
			if( (char)c == 27 )
			{ break; }
		}


		cv::Mat cdst;

		/*  Line detection
		 * white line on a black/gray background
		 * horizontal
		 * a minimum of vMinLine should be continuous and exposed,
		 * the line is extended to the limits of the image for evaluation porporsues
		 * start and end points of the line are found (extended line till there is no white )
		 */



		getScreenResolution(width, height);
		trackbar_window_name="Trackbar Hough";
		cv::namedWindow( trackbar_window_name, cv::WINDOW_NORMAL );
		cv::resizeWindow(trackbar_window_name, width/3, height);
		cv::moveWindow	(	trackbar_window_name,0,0) ;



		int const max_value_rho = 255;

		/// Create Trackbar to choose type of Threshold
		std::string trackbar_rho = "rho/10";
		cv::createTrackbar( trackbar_rho,
				trackbar_window_name, &rho,
				max_value_rho, voidMethod);


		std::string trackbar_theta = "thetagrad";
		int const max_value_theta = 360;
		cv::createTrackbar( trackbar_theta,
				trackbar_window_name, &theta,
				max_value_theta, voidMethod );


		std::string trackbar_threshold = "threshold";
		int const max_threshold = 10000;
		cv::createTrackbar( trackbar_threshold,
				trackbar_window_name, &threshold,
				max_threshold, voidMethod );

		std::string trackbar_minLineLength = "minLineLength";
		int const max_minLineLength = 1000;
		cv::createTrackbar( trackbar_minLineLength,
				trackbar_window_name, &minLineLength,
				max_minLineLength, voidMethod );

		std::string trackbar_maxLineGap = "maxLineGap";
		int const max_maxLineGap = 1000;
		cv::createTrackbar( trackbar_maxLineGap,
				trackbar_window_name, &maxLineGap,
				max_maxLineGap, voidMethod  );


		/// Create actions
		const int voidMax2=9;
		cv::createTrackbar( "action", trackbar_window_name,
				&stepView, voidMax2,
				houghLunch );

		/// Call the function to initialize
		houghLunch( 0, 0 );

		/// Wait until user finishes program
		while(true)
		{
			int c;
			c = cv::waitKey( 20 );
			if( (char)c == 27 )
			{ break; }
		}

	}
	else
	{


		std::string rawImgAdd;
		std::cout << argc<<std::endl;
		if (argc <2)
		{
			std::cout << "************Error: no option included." <<std::endl;
			std::cout<<"Add the location of the image without  \".JPG\" or  \"-tune \" to fine tuning the parameters."<<std::endl;
			std::cout<<	" -------NOW opening the default image" << std::endl;
			rawImgAdd="data/Situation_3_Auswahl/Situation_3_fern";
		}
		else
			rawImgAdd= argv[1];

		int width,height;
		getScreenResolution(width, height);
		int windowHeight=height/3;
		int windowWidth=windowHeight*1.5;

		cv::Point location1=cv::Point(2*width/3,0);
		cv::Point location2=cv::Point(2*width/3,height/2);

		cv::RotatedRect lineRotatedRect;
		Threshold_Canny(rawImgAdd,"cannyLunch1",location1,windowWidth,windowHeight);
		Threshold_HoughLinesP(rawImgAdd,"houghLunch1",location2,windowWidth,windowHeight, lineRotatedRect);


		// Analyze results
		std::vector <std::vector<cv::Point2f>>  vec_central_lines;

		for (auto singleContour:contours)
		{
			cv::Point2f tcentral_lines[2];
			std::vector<cv::Point2f> tVecCentral_lines;
			getCentralLine(singleContour,tcentral_lines);
			for (auto line:tcentral_lines)
			{
				tVecCentral_lines.push_back(line);
			}
			vec_central_lines.push_back(tVecCentral_lines);
		}



		// draw in the raw image
		cv::Mat raw;
		try {
			raw=cv::imread( rawImgAdd +".JPG"  );
		} catch (const std::exception& e) {
			std::cout << "************Error: no wrong image path." <<std::endl;
			std::cout<<"Add the location of the image without  \".JPG\""<<std::endl;
			std::cout<<	" -------NOW opening the default image" << std::endl;
			rawImgAdd="data/Situation_3_Auswahl/Situation_3_fern";
			raw=cv::imread( rawImgAdd +".JPG"  );

		}

		if (raw.empty())
		{
			std::cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		cols= raw.cols;
		rows= raw.rows;
		cv::Mat drawingOnRaw = raw.clone();
		std::vector<cv::RotatedRect> minCarRect;
		DrawRotatedRect(contours, drawingOnRaw,minCarRect);
		DrawLine(line_points, drawingOnRaw);

		// min distance to line

		double inc=0;
		float k=0;
		getLineEqu(line_points[0],line_points[1], inc,k);
		std::vector<int> distance;
		int tdistance;
		std::vector<cv::Point>closesPoint;
		cv::Point tclosesPoint;


		for (auto minareacar:minCarRect)
		{
			cv::Point2f vtx[4];
			minareacar.points(vtx);
			int dist=10000;
			for(auto points:vtx)
			{
				cv::Point2f closestpointline;
				closestpointline.x=points.x;
				closestpointline.y=k+points.x*inc;
				int newdist= euclideanDist(points,closestpointline );
				if (newdist<dist){
					tdistance=newdist;
					tclosesPoint=points;
					dist=newdist;
				}

			}
			distance.push_back(tdistance);
			closesPoint.push_back(tclosesPoint);
		}


		//draw results
		DrawLine(vec_central_lines, drawingOnRaw);
		std::cout<<"line: "<<lineRotatedRect.angle<<std::endl;
		std::vector<int> carAngles;

		for (std::size_t i=0; i<minCarRect.size();i++)
		{
			std::cout<<"car:  "<<minCarRect.at(i).angle<<std::endl;
			int angle=minCarRect.at(i).angle-lineRotatedRect.angle;
			carAngles.push_back(angle);
			std::cout<<angle<<std::endl;
			cv::RNG rng(12345);
			cv::Scalar color = cv::Scalar(0,0,0);
			cv::putText(drawingOnRaw,std::to_string(angle)+"deg",minCarRect.at(i).center,CV_FONT_HERSHEY_SIMPLEX,5,color,20);

			//distance

			cv::putText(drawingOnRaw,std::to_string(distance.at(i))+"px",closesPoint.at(i),CV_FONT_HERSHEY_SIMPLEX,5,color,20);

		}









		ShowAndSave("drawingOnRaw", cols, rows, drawingOnRaw, baseDir,1);


		std::string msg= "[   {      \"Box\": [         1286,         802,         2054,         809,        1362,         2545,         2033,         2566      ],      \"Distance To line\": 0,      \"Error Angle\": -10,      \"Next Car\": [         {            \"Distance\": 200,            \"Closer points\": [               2033,               2335,               2241,               2352            ]         }      ]   },   {      \"Box\": [         2497,         788,         3154,         809,         2222,         2490,         2863,         2635      ],      \"Distance To line\": 10,      \"Error Angle\": 20,      \"Next Car\": [         {            \"Distance\": 200,            \"Closer points\": [               2033,               2335,               2241,               2352            ]         },         {            \"Distance\": 50,            \"Closer points\": [               3189,               1003,               3258,               1003            ]         }      ]   }]";
		std::ofstream outputJsonResult;
		std::string result = baseDir + "result.json";
		outputJsonResult.open(&result[0], std::ofstream::out | std::ofstream::trunc);
		outputJsonResult<< msg;
		outputJsonResult.close();
	}

	cv::destroyAllWindows(); //destroy the created window


	std::cout << "+++END Drone_Parking_Evaluation 2!!!" << std::endl;
	return 0;



}



