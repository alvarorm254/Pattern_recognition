//g++ Filtro_video.cpp -o app `pkg-config --cflags --libs opencv`
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
  VideoCapture cap("PadronAnillos_01.avi");
  if(!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
  //Declare variables
  Mat frame;
  Mat edges;
  Mat gray;
  Mat ellipse_shape;
  vector<vector<Point>>contours;
  vector<Point>center(contours.size());
  Point2f rect_points[4];
  
  while(1)
  {
     
    //read & verify
    cap>>frame;
    if(frame.empty())
      break;
    
    //to gray
    cvtColor(frame,gray,cv::COLOR_RGB2GRAY);

    //to edges
    //Canny(gray,edges,200,200,3);
    threshold(gray,edges,100,250,THRESH_BINARY);
    
    //get the shape of ellipse and search
    ellipse_shape=getStructuringElement(MORPH_ELLIPSE,Size(4,4),Point(1,1));
    dilate(edges,edges,ellipse_shape);
    
    //find the contours of ellipses, store all points in two level hierachy
    contours.clear();
    findContours(edges,contours,CV_RETR_CCOMP,CHAIN_APPROX_NONE,Point(0,0));
    
    //clear and resize center
    center.resize(contours.size(),Point(0,0));

    //find the rotated rectangles and ellipses for each contour VALID
    vector<RotatedRect>minRect(contours.size()); //TODO: not re-declare (not urgent)
    vector<RotatedRect>minEllipse(contours.size()); //TODO: not re-declare (not urgent)
    int count=0;
    for(int i=0;i<contours.size();++i) //TODO: to iterators (not urgent)
    {
        if(contours[i].size()>16 && contours[i].size()<256) //ex-ante validation
        {
            minEllipse[count]=fitEllipse(Mat(contours[i]));
            minRect[count]=minAreaRect(Mat(contours[i]));
            ++count;
        }
    }
	
	//find the center
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
                    ellipse(frame,minEllipse[i],Scalar(0,255,0),1,LINE_AA);
                    break;
                }
                //TODO: if not match delete, for tracking
    
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
