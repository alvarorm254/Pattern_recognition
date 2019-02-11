//g++ main.cpp -o app `pkg-config --cflags --libs opencv` -fopenmp
#include <iostream>
#include <string>
#include <stdlib.h>

#include<omp.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
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
	VideoCapture cap("../cam1/anillos2.mp4");
	//VideoCapture cap("../cam2/anillos.avi");
	if(!cap.isOpened())
		return -1;

	//Declare variables
	int nuf = 12;
	vector<Point2f> arrange,int_arrange,org_arrange;
	vector<vector<Point2f>> int_points,fin_points;
	vector<Mat> int_frames,fin_frames;
	Mat cameraMatrix,distCoeffs;
	Mat rview,output;
	Mat lambda(3,3,CV_32FC1);
	vector<Point2f> corners;
	corners.push_back(Point2f(2000, 2000)); //min
	corners.push_back(Point2f(0, 0)); //max
	double error,AcErr=0;
	bool retracking=1;
	double tda=0;

	char text[40];

	double start_time;

	Mat frame;
	int num_frame=0;
	char c;
	
	cap >> frame;

	Size patternsize(COLS, ROWS);
	Size imageSize = frame.size();
	Mat bg_density(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));
	Mat bg_normalizado(frame.rows, frame.cols, CV_8UC3, Scalar(255, 255, 255));

	while(1)
	{
		//read and verify
		cap>>frame;
		if (frame.empty())
			break;
		num_frame++;
		
		detect_rings(frame,int_points,retracking,corners,int_frames,arrange);
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
	cout<<"\nfin_points: "<<fin_points.size();

	double rms = calibrate_function(patternsize, imageSize, 44, cameraMatrix, distCoeffs, fin_points);

	cout<<fin_frames.size();
	for (size_t i=0;i<60;++i)
	{
		fin_points.clear();
		for (size_t f=0;f<fin_frames.size();++f)
		{
			org_arrange.clear();
			arrange.clear();
			undistort(fin_frames[f],rview,cameraMatrix,distCoeffs);
			vector<Point2f> points=get_keypoints(rview);
			if(points.size()==20)
			{
				first_function(points,org_arrange);
				frontImageRings(lambda,rview,output,org_arrange,patternsize);
				vector<Point2f> points2=get_keypoints(output,1);
				if(points2.size()==20)
				{
					first_function_fp(points2,arrange);
					distortPoints(arrange,cameraMatrix,distCoeffs,lambda,patternsize);
					distortPoints(org_arrange,cameraMatrix,distCoeffs,lambda,patternsize);
					int_arrange=RectCorners(arrange,patternsize);
					//cout<<endl<<endl<<endl;
					/*for(int i=0;i<(int)arrange.size();++i)
						cout<<i<<" "<<arrange[i].x<<" : "<<arrange[i].y<<endl;
					for(int i=0;i<(int)org_arrange.size();++i)
						cout<<i<<" "<<org_arrange[i].x<<" : "<<org_arrange[i].y<<endl;*/
					if(arrange.size()==20)
					{
						for(int i=0;i<(int)arrange.size();++i)
							arrange[i]=(org_arrange[i]+arrange[i]+int_arrange[i])/3;
						fin_points.push_back(arrange);
					}
				}
			}
		}
		//cout<<fin_points.size()<<'\n';
		rms = calibrate_function(patternsize,imageSize,44,cameraMatrix,distCoeffs,fin_points);
		cout<<"error "<<i<<": "<<rms<<endl;
	}
	rms = calibrate_function(patternsize,imageSize,44,cameraMatrix,distCoeffs,fin_points,1);

	//convertPointsToHomogeneous(points2d, points3d);
	//projectPoints(points3d, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), camera_matrix, dist_coeffs, distorted_points2d);

	destroyAllWindows();

	//Read video (or camera) & test
	VideoCapture cap2("../cam2/anillos.avi");
	//VideoCapture cap2("../cam1/anillos2.mp4");
	if(!cap2.isOpened())
		return -1;

	cap2 >> frame;
	
	int_points.clear();
	retracking=1;
	int_frames.clear();
	arrange.clear();
	
	while(1)
	{
		cap2>>frame;
		if (frame.empty())
			break;
		Mat temp=frame.clone();
		undistort(temp,frame,cameraMatrix,distCoeffs);
		imshow("output", frame );
		vector<Point2f> points=get_keypoints(frame);
		if(points.size()==20)
		{		
			first_function(points,arrange);
			if(arrange.size()==20)
			{
				frontImageRings(lambda,frame,rview,arrange,patternsize);
				resize(rview,rview,Size(),0.5,0.5,CV_INTER_CUBIC);
				imshow("fronto", rview );
			}
		}
		c=(char)waitKey(20);
    		if(c == 27)
			break;
	}

	return 0;
}
