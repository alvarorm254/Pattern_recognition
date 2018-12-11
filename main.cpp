//g++ main.cpp -o app `pkg-config --cflags --libs opencv`
#include <iostream>
#include <string>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>

#include <cmath>

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
  
  while(1)
  {
    //clear variables
    contours.clear();
	center_v.clear();

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
    dilate(edges,edges,ellipse_shape);
    erode(edges,edges,ellipse_shape);
    
    //find the contours of ellipses
    findContours(edges,contours,CV_RETR_CCOMP,CHAIN_APPROX_NONE,Point(0,0));
	//TODO: use hierachy and delete all that not have hierachy

    //find the rotated rectangles and ellipses for each contour VALID
	cout<<contours.size()<<" ";
    minRect.resize(contours.size());
    minEllipse.resize(contours.size());
    int count=0;
    for(int i=0;i<contours.size();++i)
    {
        if(contours[i].size()>16 && contours[i].size()<256) //ex-ante validation
        {
            minEllipse[count]=fitEllipse(Mat(contours[i]));
            minRect[count]=minAreaRect(Mat(contours[i]));
            ++count;
        }
    }
	
	//find the center
	center.resize(contours.size(),Point(0,0));
    for(int i=0;i<count;++i)
    {
        minRect[i].points(rect_points);
        center[i]=Point((rect_points[0].x+rect_points[2].x)/2,(rect_points[0].y+rect_points[2].y)/2); 
    }

    //ex-post validation, distance of center heuristic
    for(int i=0;i<count;++i)
        for(int j=0;j<count;++j)
            if(j!=i)
                if(sqrt(pow((center[i].x-center[j].x),2)+pow((center[i].y-center[j].y),2))<4)
                {
		    ellipse(frame,minEllipse[i],Scalar(0,255,0),1,CV_AA);
		    circle(frame,center[i],3.0,Scalar( 0,255,0),-1, 8 );
		    center_v.push_back(center[i]);
                    break;
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
