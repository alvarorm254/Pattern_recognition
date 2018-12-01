#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/plot.hpp>

#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
  VideoCapture cap("PadronAnillos_01.avi");
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
    // Display the resulting frame
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
