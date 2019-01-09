//g++ main.cpp -o app `pkg-config --cflags --libs opencv` -fopenmp
#include <iostream>
#include <string>
#include <stdlib.h>

#include<omp.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "integral_threshold.h"
//#include <opencv2/plot.hpp>

#include <cmath>

#define ROWS 3
#define COLS 4
#define NUM_RINGS ROWS*COLS
#define NUM_ELLIPSES ROWS*COLS*2
#define THREAD_COUNT 8
#define MIN_ELLIPSE_POINTS 24
#define MAX_ELLIPSE_POINTS 256
#define MAX_DISTANCE_CENTERS 4

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	//Read video (or camera) & test
	VideoCapture cap("../padron1.avi");
	if(!cap.isOpened())
		return -1;

	//Declare variables
	Mat frame,edges,gray,gaussian; //ellipse_shape=getStructuringElement(MORPH_ELLIPSE,Size(2,2),Point(1,1));
	vector<vector<Point>>contours;
	vector<Point>center,center_v,center_loss,center_aux;
	Point center1,center2,mass_center;
	//vector<RotatedRect>minEllipse;//minRect;
	RotatedRect minRect1,minRect2;
	Point2f rect_points1[4],rect_points2[4];
	vector<Vec4i> hierarchy;
	int i,aux,aux2,max_radio,count_all,num_frame=0;
	vector<int> finded(NUM_RINGS,0);

	char text[40];

	double start_time;

	int all=0,full=0,retracks=0;

	char c=(char)waitKey(3000);


	while(1)
	{
		//clear variables
		center_loss.clear();
		center_aux.clear();

		//read and verify
		cap>>frame;
		if(frame.empty())
			break;
		num_frame++;

		sprintf(text,"Frame: %d",num_frame);
		putText(frame,text,Point2f(15,20),FONT_HERSHEY_PLAIN,1.25,Scalar(0,0,255,255),1);

		start_time=omp_get_wtime();

		//to gray
		cvtColor(frame,gray,cv::COLOR_RGB2GRAY);

		//gaussian blur
		GaussianBlur(gray,gaussian,Size(3,3),0,0);

		//to edges
		//threshold(gaussian,edges,100,255,THRESH_BINARY);
		edges=integral_threshold(gaussian,0.85);

		//find the contours of ellipses
		contours.clear();
		hierarchy.clear();
		findContours(edges,contours,hierarchy,CV_RETR_TREE,CHAIN_APPROX_NONE,Point(0,0)); //can change methods
		//hierarchy [Next, Previous, First_Child, Parent]

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
				{
					//minEllipse.push_back(fitEllipse(Mat(contours[i])));
					//ellipse(frame,minEllipse.back(),Scalar(0,255,0),1,CV_AA);
					//minEllipse.push_back(fitEllipse(Mat(contours[hierarchy[i][2]])));
					//ellipse(frame,minEllipse.back(),Scalar(0,255,0),1,CV_AA);
					center.push_back(Point((center1.x+center2.x)/2,(center1.y+center2.y)/2));
					//circle(frame,center.back(),3.0,Scalar(0,255,0),-1,8);
				}
			}

		//delete leftover rings
		while(center.size()>NUM_RINGS && center.size()!=0)
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
			//minEllipse.erase(minEllipse.begin()+(2*aux)); //TODO: falta probar no es necesario
			//minEllipse.erase(minEllipse.begin()+(2*aux+1)); //TODO: falta probar no es necesario
		}

		//print the id-number of the ring
		for(i=0;i<(int)center.size();i++)
		{
			circle(frame,center[i],3.0,Scalar(0,255,0),-1,CV_AA);
			sprintf(text,"%d",i);
			putText(frame,text,center[i],FONT_HERSHEY_PLAIN,2,Scalar(0,0,255,255),2);
		}

		sprintf(text,"Rings: %d",(int)center.size());
		putText(frame,text,Point2f(15,40),FONT_HERSHEY_PLAIN,1.25,Scalar(0,0,255,255),1);


		//TODO: using bounding boxes for teletransportation of bad friend

		//TODO: for no rastered points, use the moviment of another points to predict the future location

		//put in rame some important data

		if(center.size()==NUM_RINGS)
			full++;
		all++;

		sprintf(text,"Time: %.3f",omp_get_wtime()-start_time);
		putText(frame,text,Point2f(15,60),FONT_HERSHEY_PLAIN,1.25,Scalar(0,0,255,255),1);

		sprintf(text,"acc: %.2f",100.0*(float)full/float(all));
		putText(frame,text,Point2f(15,80),FONT_HERSHEY_PLAIN,1.25,Scalar(0,0,255,255),1);

		//TODO TODO: tracking of points based on the center and tracing lines
		imshow( "Frame", frame );

		// Press  ESC on keyboard to exit
		char c=(char)waitKey(1);
		if(c==27)
			break;

		if(num_frame==2310)
			cin>>c;
	}

	cout<<"\nNumber of frames: "<<all;

	cout<<"\nPrecision:"<<100.0*(float)full/float(all);

	cout<<"\nRe-track:"<<retracks;

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();

	return 0;
}
