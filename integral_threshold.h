#include <opencv2/opencv.hpp> //Include file for every supported OpenCV function

using namespace cv;

Mat integral_threshold(Mat input, float threshold){
	Size dim = input.size();
	Mat output(dim, CV_8UC1);
	unsigned int integral_img[dim.height][dim.width];
	//s.height
	unsigned long sum;
	int count;
	int xmin,xmax,ymin,ymax;

	
	for (int i=0; i<dim.width; i++)
	{
		// reset this column sum
		sum = 0;
		for (int j=0; j<dim.height; j++)
		{
			sum += input.at<unsigned char>(j,i);
			if (i==0)
				integral_img[j][i] = sum;
			else
				integral_img[j][i] = integral_img[j][i-1] + sum;
		}
	}
	int r=(int) dim.width/(8*2);/////////////////////////////////////
		// perform thresholding
	for (int i=0; i<dim.width; i++)
	{
		for (int j=0; j<dim.height; j++)
		{
			// set the SxS region
			xmin=i-r; xmax=i+r;
			ymin=j-r; ymax=j+r;

			// check the border
			if (xmin < 0) xmin = 0;
			if (xmax >= dim.width) xmax = dim.width-1;
			if (ymin < 0) ymin = 0;
			if (ymax >= dim.height) ymax = dim.height-1;
			
			count = (xmax-xmin)*(ymax-ymin);///////////////////////////////////////////

			sum = integral_img[ymax][xmax]-
				  integral_img[ymin][xmax]-
				  integral_img[ymax][xmin]+
				  integral_img[ymin][xmin];

			if (input.at<uchar>(j,i)<uchar(32) || (unsigned long)(input.at<uchar>(j,i))*count < (sum*(threshold)))
				output.at<uchar>(j,i) = uchar (0);
			else
				output.at<uchar>(j,i) = uchar (255);
		}
	}
	return output;
}
