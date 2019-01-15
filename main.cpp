//g++ main.cpp -o app `pkg-config --cflags --libs opencv` -fopenmp
#include <iostream>
#include <string>
#include <stdlib.h>

#include<omp.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "integral_threshold.h"
#include "rings_functions.h"
#include "iterativo.h"
//#include <opencv2/plot.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <vector>
#include <math.h>
#include <fstream>

#include <cmath>

#define ROWS 4
#define COLS 5
#define NUM_RINGS ROWS*COLS
#define NUM_ELLIPSES ROWS*COLS*2
#define THREAD_COUNT 8
#define MIN_ELLIPSE_POINTS 16
#define MAX_ELLIPSE_POINTS 256
#define MAX_DISTANCE_CENTERS 4

using namespace std;
using namespace cv;

int main()
{
	//Read video (or camera) & test
	VideoCapture cap("../cam2/anillos.avi");
	if(!cap.isOpened())
		return -1;

	//Declare variables
	int nuf = 60;
	vector<Point2f> arrange,int_arrange,org_arrange;
	vector<vector<Point2f>>int_points,fin_points;
	vector<Mat>int_frames,fin_frames;
	Mat cameraMatrix, distCoeffs;
	Mat rview, output;
	Mat lambda(3,3,CV_32FC1);
	vector<Point2f> corners;
	corners.push_back(Point2f(2000, 2000)); //min
	corners.push_back(Point2f(0, 0)); //max
	double  error,AcErr=0;
	bool retracking=1;
	double tda=0;

	char text[40];

	double start_time;

	//Declare variables
	Mat frame,edges,gray,gaussian; //ellipse_shape=getStructuringElement(MORPH_ELLIPSE,Size(2,2),Point(1,1));
	vector<vector<Point>>contours;
	vector<Point2f>center;
	Point2f center1,center2,mass_center,last_mass_center;
	//vector<RotatedRect>minEllipse;//minRect;
	RotatedRect minRect1,minRect2;
	Point2f rect_points1[4],rect_points2[4];
	vector<Vec4i> hierarchy;
	int i,j,k,aux,aux2,num_frame=0,A,B,C;
	vector<int> finded(NUM_RINGS);
	char c;
	
	cap >> frame;

	Size patternsize(COLS, ROWS);
	Size imageSize = frame.size();
	Mat bg_density(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
	Mat bg_normalizado(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));

	RotatedRect minEllipse;

	while(1)
	{
		//read and verify
		cap>>frame;
		if (frame.empty())
			break;
		num_frame++;

		//to gray
		cvtColor(frame,gray,cv::COLOR_RGB2GRAY);

		//gaussian blur
		GaussianBlur(gray,gaussian,Size(3,3),0,0);

		//to edges
		edges=integral_threshold(gaussian,0.85);
		imshow("output2",edges);

		//find the contours of ellipses
		contours.clear();
		hierarchy.clear();
		findContours(edges,contours,hierarchy,CV_RETR_TREE,CHAIN_APPROX_NONE,Point(0,0)); //can change methods
		//hierarchy [Next, Previous, First_Child, Parent]

		//early skip frame
		if(contours.size()<NUM_RINGS*2)
		{	
			retracking=1;
			imshow("output",frame);
			continue;
		}

		//"delete" (HYERARCHY[2],[3]=-1) contours by size **MIN_ELLIPSE_POINTS < size < MAX_ELLIPSE_POINTS**
		for(i=0;i<(int)contours.size();++i)
			if(contours[i].size()>MAX_ELLIPSE_POINTS)
			{
				//eliminate in childs
				aux=hierarchy[i][2];
				while(aux!=-1)
				{
					hierarchy[aux][3]=hierarchy[i][3];
					aux=hierarchy[aux][0];
				}
				//eliminate in father
				if(hierarchy[hierarchy[i][3]][2]==i)
					hierarchy[hierarchy[i][3]][2]=hierarchy[i][0];
				hierarchy[i][2]=-1;
				hierarchy[i][3]=-1;
			}
			else
				if(contours[i].size()<MIN_ELLIPSE_POINTS)
				{
					hierarchy[i][2]=-1;
					//eliminate in father
					if(hierarchy[i][3]!=-1)
					{
						if(hierarchy[hierarchy[i][3]][2]==i)
							hierarchy[hierarchy[i][3]][2]=hierarchy[i][0];
						hierarchy[i][3]=-1;
					}
				}

		//"delete" (HYERARCHY[2],[3]=-1) father and childs, when the number of childs is bigger than 1
		for(i=0;i<(int)contours.size();++i)
			if(hierarchy[i][3]!=-1 && (hierarchy[i][0]!=-1 || hierarchy[i][1]!=-1))
			{
				hierarchy[hierarchy[i][3]][2]=-1;
				hierarchy[hierarchy[i][3]][3]=-1;
				hierarchy[i][2]=-1;
				hierarchy[i][3]=-1;
			}

		//Calculate centers
		//minEllipse.clear();
		center.clear();
		for(i=0;i<(int)contours.size();++i)
			if(hierarchy[i][2]!=-1)
			{
				minRect1=minAreaRect(Mat(contours[i]));
				minRect2=minAreaRect(Mat(contours[hierarchy[i][2]]));
				minRect1.points(rect_points1);
				minRect2.points(rect_points2);
				center1=Point((rect_points1[0].x+rect_points1[2].x)/2,(rect_points1[0].y+rect_points1[2].y)/2);
				center2=Point((rect_points2[0].x+rect_points2[2].x)/2,(rect_points2[0].y+rect_points2[2].y)/2);
				if(sqrt(pow((center1.x-center2.x),2)+pow((center1.y-center2.y),2))<MAX_DISTANCE_CENTERS)
					center.push_back(Point((center1.x+center2.x)/2,(center1.y+center2.y)/2));
			}

		//early skip frame		
		if(center.size()<NUM_RINGS)
		{	
			retracking=1;
			imshow("output",frame);
			continue;
		}

		//delete leftover rings
		while(center.size()>NUM_RINGS) //&& center.size()!=0)
		{
			mass_center=Point(0,0);
			for(i=0;i<(int)center.size();++i)
				mass_center+=center[i];
			mass_center.x/=center.size();
			mass_center.y/=center.size();
			aux=0;
			aux2=sqrt(pow((mass_center.x-center[0].x),2)+pow((mass_center.y-center[0].y),2));
			for(i=1;i<(int)center.size();++i)
				if(sqrt(pow((mass_center.x-center[i].x),2)+pow((mass_center.y-center[i].y),2))>aux2)
				{
					aux=i;
					aux2=sqrt(pow((mass_center.x-center[i].x),2)+pow((mass_center.y-center[i].y),2));
				}
			center.erase(center.begin()+aux);
		}

		detect_rings(frame, int_points, retracking, corners, int_frames, arrange,center);
      	imshow("output",frame);
    	
		c=(char)waitKey(1);
		if(c==27)
			break;
	}

	cap.release();
	destroyAllWindows();

	see_density(bg_density, int_points, "No_normalizado");
	normalize_density_nuf(corners, int_points, fin_points, int_frames, fin_frames, nuf);
	//get_random_samples(int_points, fin_points, int_frames, fin_frames, nuf);
	see_density(bg_normalizado, fin_points, "Normalizado");

	//write_samples(fin_frames);

	double rms = calibrate_function(patternsize, imageSize, 45.75, cameraMatrix, distCoeffs, fin_points);

	/*cout<<fin_frames.size();
	for (size_t i = 0; i < 5; i++)
	{
		fin_points.clear();
		for (size_t f = 0; f < fin_frames.size(); f++)
		{
			undistort(fin_frames[f], rview, cameraMatrix, distCoeffs);
			imwrite("output.jpg",rview);
			vector<Point2f> points = get_keypoints(rview);
			if (points.size() == 20)
			{
				first_function(points, arrange);
				frontImageRings(lambda, rview, output, arrange, patternsize);
				//imshow("output2",output);
				imwrite("output2.jpg",output);
				//vector<Point2f> points2 = core_get_keypoints(output, 500, 1, 3);
				vector<Point2f> points2 = core_get_keypoints(output, 200, 0, 1);
				if (points2.size() == 20)
				{
					cout<<"SI";
					first_function(points2, arrange);
					int_arrange = RectCorners(arrange, patternsize);
					distortPoints(arrange, cameraMatrix, distCoeffs, lambda, patternsize);
					distortPoints(int_arrange, cameraMatrix, distCoeffs, lambda, patternsize);
					vector<Point2f> points3 = get_keypoints(fin_frames[f]);
					if (points3.size() == 20)
					{
						first_function(points3, org_arrange);
						for (size_t i = 0; i < arrange.size(); i++)
						{
							arrange[i].x = (arrange[i].x + int_arrange[i].x + org_arrange[i].x)/3;
							arrange[i].y = (arrange[i].y + int_arrange[i].y + org_arrange[i].y)/3;
						}
						fin_points.push_back(arrange);
					}
				}
				else
				{
					cout<<"NO";
				}
			}
		}
		cout<<fin_points.size()<<'\n';
		double rms = calibrate_function(patternsize, imageSize, 44.3, cameraMatrix, distCoeffs, fin_points);
		cout<<rms;
	}*/

	destroyAllWindows();


	//Read video (or camera) & test
	VideoCapture cap2("../cam2/anillos.avi");
	if(!cap2.isOpened())
		return -1;

	cap2 >> frame;

	while(1)
	{
		cap2>>frame;
		if (frame.empty())
			break;
		Mat temp = frame.clone();
		undistort(temp, frame, cameraMatrix, distCoeffs);
		imshow("undistort", frame );
		c=(char)waitKey(1);
    		if(c == 27)
			break;
	}

	return 0;
}
