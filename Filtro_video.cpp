#include <iostream>
#include <string>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>

#include <cmath>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[]) {
  VideoCapture cap("calibration_kinectv2.avi");
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  while(1){
    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    //IplImage* gray = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
    cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
    Mat edges;
    int param1=strtol(argv[1],NULL,10);
    int param2=strtol(argv[2],NULL,10);
    int param3=strtol(argv[3],NULL,10);
    int scale = 1;
    int delta = 0;
    //Canny(frame, edges, param1, param2, param3, false);
    //Mat grad_x, abs_grad_x;
    //int ddepth = CV_16S;
    //Sobel( frame, grad_x, ddepth,param1, param2, param3, scale, delta, BORDER_DEFAULT );
    //Sobel( frame, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    //convertScaleAbs( grad_x, abs_grad_x );
    //GaussianBlur( frame, frame, Size( 15, 15 ), 0, 0 );
    Canny(frame, edges, param1, param2, param3, false);
    // Display the resulting frame
    imshow( "Frame", edges );
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
