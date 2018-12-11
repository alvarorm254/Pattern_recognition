//g++ main.cpp -o app `pkg-config --cflags --libs opencv`
#include <iostream>
#include <string>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>

#include <cmath>

#define NUM_ELLIPSES 30

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	//Read video (or camera) & test
	VideoCapture cap("video1.avi");
	if(!cap.isOpened())
		return -1;

	//Declare variables
	Mat frame,edges,gray,gaussian,ellipse_shape;
	vector<vector<Point>>contours;
	vector<Point>center,center_v;
	vector<RotatedRect>minRect,minEllipse;
	Point2f rect_points[4];
	vector<Vec4i> hierarchy;
	int i,j,count,next,previous;

	while(1)
	{
		//clear variables
		contours.clear();
		center_v.clear();
		hierarchy.clear();

		//read and verify
		cap>>frame;
		if(frame.empty())
			break;

		//to gray
		cvtColor(frame,gray,cv::COLOR_RGB2GRAY);

		//gaussian blur
		GaussianBlur(gray,gaussian,Size(3,3),0,0);

		//to edges
		threshold(gaussian,edges,100,255,THRESH_BINARY);
		//TODO: integral threshold

		//dilate and erode for ellipses
		ellipse_shape=getStructuringElement(MORPH_ELLIPSE,Size(2,2),Point(1,1));
		erode(edges,edges,ellipse_shape);
		dilate(edges,edges,ellipse_shape);

		//find the contours of ellipses
		findContours(edges,contours,hierarchy,CV_RETR_TREE,CHAIN_APPROX_NONE,Point(0,0)); //can change methods
		//hierarchy [Next, Previous, First_Child, Parent]

		//filter contours TODO: improve performance
		for(i=0;i<contours.size();++i)
			if(contours[i].size()>256)
				for(j=0;j<contours.size();++j)
					if(hierarchy[j][3]==i)
						hierarchy[j][3]=-1;
		for(i=0;i<contours.size();++i)
			if(contours[i].size()<24)
				if(hierarchy[i][3]!=-1)
					if(hierarchy[i][0]!=-1)
						hierarchy[hierarchy[i][3]][2]=hierarchy[i][0];
					else
						hierarchy[hierarchy[i][3]][2]=hierarchy[i][1];

		for(i=0;i<contours.size();++i)
			if(contours[i].size()<24 || (hierarchy[i][2]==-1 && hierarchy[i][3]==-1) || contours[i].size()>256)
			{
				contours.erase(contours.begin()+i);
				hierarchy.erase(hierarchy.begin()+i);
				--i;
			}
		count=contours.size();

		char str[3];
		sprintf(str,"%d",count);
		putText(frame,str,Point2f(15,25),FONT_HERSHEY_PLAIN,2,Scalar(0,0,255,255));

		//find the rotated rectangles and ellipses for each contour VALID
		minRect.resize(count);
		minEllipse.resize(count);
		for(i=0;i<count;++i)
		{
			minEllipse[i]=fitEllipse(Mat(contours[i]));
			minRect[i]=minAreaRect(Mat(contours[i]));
		}

		//find the center
		center.resize(count,Point(0,0));
		for(i=0;i<count;++i)
		{
			minRect[i].points(rect_points);
			center[i]=Point((rect_points[0].x+rect_points[2].x)/2,(rect_points[0].y+rect_points[2].y)/2);
		}

		//ex-post validation, distance of center heuristic
		for(i=0;i<count;++i)
		{
			for(j=0;j<count;++j)
				if(j!=i)
					if(sqrt(pow((center[i].x-center[j].x),2)+pow((center[i].y-center[j].y),2))<4)
					{
						ellipse(frame,minEllipse[i],Scalar(0,255,0),1,CV_AA);
						circle(frame,center[i],3.0,Scalar( 0,255,0),-1, 8 );
						center_v.push_back(center[i]);
						break;
					}
		}

		for(i=0;i<count;++i)
		{
			char str2[3];
			sprintf(str2,"%d",i);
			putText(frame,str2,contours[i][0],FONT_HERSHEY_PLAIN,0.5,Scalar(0,0,255,255));
		}


		//TODO: if not match delete, for tracking
		Rect rect = boundingRect(center_v);
		rectangle( frame, rect.tl(), rect.br(), Scalar( 0,255,0), 2, 8, 0 );

		//TODO TODO: tracking of points based on the center and tracing lines
		imshow( "Frame", frame );

		// Press  ESC on keyboard to exit
		char c=(char)waitKey(25);
		if(c==27)
			break;
	}

	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();
	return 0;
}
