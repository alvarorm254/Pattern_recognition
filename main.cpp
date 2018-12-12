//g++ main.cpp -o app `pkg-config --cflags --libs opencv`
#include <iostream>
#include <string>
#include <stdlib.h>

#include<omp.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include "integral_threshold.h"
//#include <opencv2/plot.hpp>

#include <cmath>

#define NUM_ELLIPSES 12
#define THREAD_COUNT 8

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	//Read video (or camera) & test
	VideoCapture cap("../padron1.avi");
	if(!cap.isOpened())
		return -1;

	//Declare variables
	Mat frame,edges,gray,gaussian,ellipse_shape=getStructuringElement(MORPH_ELLIPSE,Size(2,2),Point(1,1));
	vector<vector<Point>>contours;
	vector<Point>center,center_v,center_v2,center_aux;
	vector<RotatedRect>minRect,minEllipse;
	Point2f rect_points[4];
	vector<Vec4i> hierarchy;
	int i,j,k,count,aux;
	Point mean_center;

	double start_time;

	while(1)
	{
		//clear variables
		contours.clear();
		center_v2.clear();
		center_aux.clear();
		hierarchy.clear();

		//read and verify
		cap>>frame;
		if(frame.empty())
			break;

		start_time=omp_get_wtime();

		//to gray
		cvtColor(frame,gray,cv::COLOR_RGB2GRAY);

		//gaussian blur
		GaussianBlur(gray,gaussian,Size(3,3),0,0);

		//to edges
		//threshold(gaussian,edges,100,255,THRESH_BINARY);
		edges=integral_threshold(gaussian,0.85);

		//dilate and erode for ellipses
		erode(edges,edges,ellipse_shape);
		dilate(edges,edges,ellipse_shape);

		//find the contours of ellipses
		findContours(edges,contours,hierarchy,CV_RETR_TREE,CHAIN_APPROX_NONE,Point(0,0)); //can change methods
		//hierarchy [Next, Previous, First_Child, Parent]

		//filter contours by size and HIERARCHY AND SIZE
		for(i=0;i<(int)contours.size();++i)
			if(contours[i].size()>256)
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
				if(contours[i].size()<16)
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

		//delete false positive contours
		for(i=0;i<(int)contours.size();++i)
			if(contours[i].size()<24 || (hierarchy[i][2]==-1 && hierarchy[i][3]==-1) || contours[i].size()>256)
			//TODO: verificar || (hierarchy[i][2]==-1 && (hierarchy[i][0]!=-1 || hierarchy[i][1]!=-1)
			{
				contours.erase(contours.begin()+i);
				hierarchy.erase(hierarchy.begin()+i);
				--i;
			}
		count=contours.size();

		//find the rotated rectangles, ellipses and centers
		minRect.resize(count);
		minEllipse.resize(count);
		center.resize(count);
		for(i=0;i<count;++i)
		{
			minEllipse[i]=fitEllipse(Mat(contours[i]));
			minRect[i]=minAreaRect(Mat(contours[i]));
			minRect[i].points(rect_points);
			center[i]=Point((rect_points[0].x+rect_points[2].x)/2,(rect_points[0].y+rect_points[2].y)/2);
		}

		//TODO: delete when count>NUM_ELLIPSES, hierarchy can help

		//TODO: for no rastered points, use the moviment to predict the future location

		//ex-post validation, distance of center heuristic
		for(i=0;i<count;++i)
			for(j=0;j<count;++j)
				if(j!=i)
					if(sqrt(pow((center[i].x-center[j].x),2)+pow((center[i].y-center[j].y),2))<4)
					{
						ellipse(frame,minEllipse[i],Scalar(0,255,0),1,CV_AA);
						circle(frame,center[i],3.0,Scalar(0,255,0),-1,8);

						ellipse(frame,minEllipse[j],Scalar(0,255,0),1,CV_AA);
						circle(frame,center[j],3.0,Scalar(0,255,0),-1,8);

						mean_center=Point((center[i].x+center[j].x)/2,(center[i].y+center[j].y)/2);

						aux=0;
						for(k=0;k<(int)center_v.size();++k)
							if(sqrt(pow((center_v[k].x-mean_center.x),2)+pow((center_v[k].y-mean_center.y),2))<8)
							{
								center_v[k]=mean_center;
								aux=1;
								break;
							}
						if(aux==0)
							center_v2.push_back(mean_center);

						center_aux.push_back(mean_center);

						if(i<j)
						{
							center.erase(center.begin()+j);
							minEllipse.erase(minEllipse.begin()+j);
							minRect.erase(minRect.begin()+j);

							center.erase(center.begin()+i);
							minEllipse.erase(minEllipse.begin()+i);
							minRect.erase(minRect.begin()+i);
						}
						else
						{
							center.erase(center.begin()+i);
							minEllipse.erase(minEllipse.begin()+i);
							minRect.erase(minRect.begin()+i);

							center.erase(center.begin()+j);
							minEllipse.erase(minEllipse.begin()+j);
							minRect.erase(minRect.begin()+j);
							i--;
						}

						i--;
						count=count-2;

						break;
					}

		if(center_v.size()==0)
			center_v=center_v2;

		if(center_v2.size()>3)
			center_v=center_aux;
		else
			if(center_v.size()<12)
				center_v.insert(center_v.end(),center_v2.begin(),center_v2.end());
		//put in rame some important data
		for(i=0;i<(int)center_v.size();++i)
		{
			char str2[3];
			sprintf(str2,"%d",i);
			putText(frame,str2,center_v[i],FONT_HERSHEY_PLAIN,2,Scalar(0,0,255,255),2);
		}

		char str[10];
		sprintf(str,"%f",omp_get_wtime()-start_time);
		putText(frame,str,Point2f(15,25),FONT_HERSHEY_PLAIN,1.5,Scalar(0,0,255,255),2);

		//TODO TODO: tracking of points based on the center and tracing lines
		imshow( "Frame", frame );

		// Press  ESC on keyboard to exit
		char c=(char)waitKey(1);
		if(c==27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();

	return 0;
}
