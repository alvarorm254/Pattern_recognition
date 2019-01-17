//g++ -ggdb facedetect.cpp -o facedetect `pkg-config --cflags --libs opencv`
//#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void frontImageRings( Mat &lambda,Mat rview,Mat &output,vector<Point2f> &corners,Size patternsize){
  Point2f inputQuad[4];
  Point2f outputQuad[4];

  /*inputQuad[0]=corners[15];
  inputQuad[1]=corners[19];
  inputQuad[2]=corners[4];
  inputQuad[3]=corners[0];
  outputQuad[0]=Point2f( 55,55 );
  outputQuad[1]=Point2f( 555-1,55);
  outputQuad[2]=Point2f( 555-1,430-1);
  outputQuad[3]=Point2f( 55,430-1);*/

  inputQuad[0]=corners[15];
  inputQuad[1]=corners[19];
  inputQuad[2]=corners[4];
  inputQuad[3]=corners[0];
  outputQuad[0]=Point2f(55,55);
  outputQuad[1]=Point2f(585,55);
  outputQuad[2]=Point2f(585,305);
  outputQuad[3]=Point2f(55,305);

  lambda=getPerspectiveTransform(inputQuad,outputQuad);
  warpPerspective(rview,output,lambda,output.size() );
  lambda=getPerspectiveTransform(outputQuad,inputQuad);
}

void distortPoints(vector<Point2f> &corners,Mat cameraMatrix,Mat distCoeffs,Mat lambda,Size patternsize) {
  double y,x,r,idistra,t;

  for( int i=0; i < patternsize.height; ++i ){
    for( int j=0; j < patternsize.width; ++j ){
      x=lambda.at <double>(0,0) * corners[j+(i*patternsize.width)].x;
      x+=lambda.at <double>(0,1) * corners[j+(i*patternsize.width)].y;
      x+=lambda.at <double>(0,2);

      y=lambda.at <double>(1,0) * corners[j+(i*patternsize.width)].x;
      y+=lambda.at <double>(1,1) * corners[j+(i*patternsize.width)].y;
      y+=lambda.at <double>(1,2);

      t=lambda.at <double>(2,0) * corners[j+(i*patternsize.width)].x;
      t+=lambda.at <double>(2,1) * corners[j+(i*patternsize.width)].y;
      t+=lambda.at <double>(2,2);

      x /= t;
      y /= t;

      x=(x-cameraMatrix.at <double>(0,2))/cameraMatrix.at <double>(0,0);
      y=(y-cameraMatrix.at <double>(1,2))/cameraMatrix.at <double>(1,1);

      r=(x*x)+(y*y);
      idistra=(1+(distCoeffs.at <double>(0,0)*r)+(distCoeffs.at <double>(0,1)*r*r)+(distCoeffs.at <double>(0,4)*r*r*r));
      x=x*idistra+((2*distCoeffs.at <double>(0,2)*x*y)+(distCoeffs.at <double>(0,3)*(r+(2*x*x))));
      y=y*idistra+((2*distCoeffs.at <double>(0,3)*x*y)+(distCoeffs.at <double>(0,2)*(r+(2*y*y))));

      corners[j+(i*patternsize.width)].x=x*cameraMatrix.at <double>(0,0)+cameraMatrix.at <double>(0,2);
      corners[j+(i*patternsize.width)].y=y*cameraMatrix.at <double>(1,1)+cameraMatrix.at <double>(1,2);
    }
  }
}

void rectaHor (vector<Point2f> corners,Size patternsize,std::vector<double> &m,std::vector<double> &b){
  double xm,ym,s1,s2,s3;
  for( int i=0; i < patternsize.height; ++i ){
    xm=0;
    ym=0;
    for( int j=0; j < patternsize.width; ++j ){
      xm+=corners[j+(i*patternsize.width)].x;
      ym+=corners[j+(i*patternsize.width)].y;
    }
    xm /= (patternsize.width);
    ym /= (patternsize.width);
    s1=0;
    s2=0;
    s3=0;
    for( int j=0; j < patternsize.width; ++j ){
      s1+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].y-ym);
      s2+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].x-xm);
      s3+=(corners[j+(i*patternsize.width)].y-ym)*(corners[j+(i*patternsize.width)].y-ym);
    }
    if(s3 > s2){
      m[i]=s1/s3;
      b[i]=xm-(m[i]*ym);
    }else{
      m[i]=s1/s2;
      b[i]=ym-(m[i]*xm);
    }
  }
}

void rectaVer (vector<Point2f> corners,Size patternsize,std::vector<double> &m,std::vector<double> &b){
  double xm,ym,s1,s2,s3;
  for( int j=0; j < patternsize.width; ++j ){
    xm=0;
    ym=0;
    for( int i=0; i < patternsize.height; ++i ){
      xm+=corners[j+(i*patternsize.width)].x;
      ym+=corners[j+(i*patternsize.width)].y;
    }
    xm /= (patternsize.height);
    ym /= (patternsize.height);
    s1=0;
    s2=0;
    s3=0;
    for( int i=0; i < patternsize.height; ++i ){
      s1+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].y-ym);
      s2+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].x-xm);
      s3+=(corners[j+(i*patternsize.width)].y-ym)*(corners[j+(i*patternsize.width)].y-ym);
    }
    if(s3 > s2){
      m[j]=s1/s3;
      b[j]=xm-(m[j]*ym);
    }else{
      m[j]=s1/s2;
      b[j]=ym-(m[j]*xm);
    }
  }
}

vector<Point2f> RectCorners (vector<Point2f> corners,Size patternsize){
  double y,x;
  std::vector<Point2f> points;
  std::vector<double> b1(patternsize.height),m1(patternsize.height),b2(patternsize.width),m2(patternsize.width);
  rectaHor(corners,patternsize,m1,b1);
  rectaVer(corners,patternsize,m2,b2);
  for (size_t i=0; i < patternsize.height; i++) {
    for (size_t j=0; j < patternsize.width; j++) {
      x=(m2[j]*b1[i]+b2[j])/(1 -(m2[j]*m1[i]));
      y=m1[i]*x+b1[i];
      points.push_back(Point2f(x,y));
    }
  }
  return points;
}

vector<Point2f> RectCornersAsime (vector<Point2f> corners,Size patternsize){
  double y,x;
  std::vector<Point2f> points1,points2,corn1,corn2,points;
  std::vector<double> b1(patternsize.height),m1(patternsize.height),b2(patternsize.height),m2(patternsize.height);

  for (size_t i=0; i < patternsize.height; i++) {
    if (i % 2 == 0) {
      for (size_t j=0; j < patternsize.width; j++) {
        corn1.push_back(corners[j+(i*patternsize.width)]);
      }
    }else{
      for (size_t j=0; j < patternsize.width; j++) {
        corn2.push_back(corners[j+(i*patternsize.width)]);
      }
    }
  }
  Size psize1(patternsize.width,corn1.size()/patternsize.width) ;
  Size psize2(patternsize.width,corn2.size()/patternsize.width) ;

  rectaHor(corn1,psize1,m2,b2);
  rectaVer(corn1,psize1,m1,b1);
  for (size_t j=0; j < psize1.height; j++) {
    for (size_t i=0; i < psize1.width; i++) {
      x=(m2[j]*b1[i]+b2[j])/(1 -(m2[j]*m1[i]));
      y=m1[i]*x+b1[i];
      points1.push_back(Point2f(x,y));
    }
  }

  rectaHor(corn2,psize2,m2,b2);
  rectaVer(corn2,psize2,m1,b1);
  for (size_t j=0; j < psize2.height; j++) {
    for (size_t i=0; i < psize2.width; i++) {
      x=(m2[j]*b1[i]+b2[j])/(1 -(m2[j]*m1[i]));
      y=m1[i]*x+b1[i];
      points2.push_back(Point2f(x,y));
    }
  }


  for (size_t i=0; i < patternsize.height; i++) {
    if (i % 2 == 0) {
      for (size_t j=0; j < patternsize.width; j++) {
        points.push_back(points1[j+((i/2)*patternsize.width)]);
      }
    }else{
      for (size_t j=0; j < patternsize.width; j++) {
        points.push_back(points2[j+((i/2)*patternsize.width)]);
      }
    }
  }
  return points;
}

float colinialidad (vector<Point2f> corners,Size patternsize){
  double  error=0,xm,ym,m,b,s1,s2,s3,y,x;
  for( int i=0; i < patternsize.height; ++i ){
    xm=0;
    ym=0;
    for( int j=0; j < patternsize.width; ++j ){
      xm+=corners[j+(i*patternsize.width)].x;
      ym+=corners[j+(i*patternsize.width)].y;
    }
    xm /= (patternsize.width);
    ym /= (patternsize.width);
    s1=0;
    s2=0;
    s3=0;
    for( int j=0; j < patternsize.width; ++j ){
      s1+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].y-ym);
      s2+=(corners[j+(i*patternsize.width)].x-xm)*(corners[j+(i*patternsize.width)].x-xm);
      s3+=(corners[j+(i*patternsize.width)].y-ym)*(corners[j+(i*patternsize.width)].y-ym);
    }
    if(s3 > s2){
      m=s1/s3;
      b=xm-(m*ym);
      for( int j=0; j < patternsize.width; ++j ){
        x=(m * corners[j+(i*patternsize.width)].y)+b;
        x -= corners[j+(i*patternsize.width)].x;
        if (m > 0.01) {
          y=(corners[j+(i*patternsize.width)].x-b) / m;
          y -= corners[j+(i*patternsize.width)].y;
          x *= x;
          y *= y;
          error+=sqrt(x * y)/sqrt(x+y);
        } else{
          x *= x;
          error+=x;
        }
      }
    }else{
      m=s1/s2;
      b=ym-(m*xm);
      for( int j=0; j < patternsize.width; ++j ){
        y=(m * corners[j+(i*patternsize.width)].x)+b;
        y -= corners[j+(i*patternsize.width)].y;
        if (m > 0.01) {
          x=(corners[j+(i*patternsize.width)].y-b) / m;
          x -= corners[j+(i*patternsize.width)].x;
          x *= x;
          y *= y;
          error+=sqrt(x * y)/sqrt(x+y);
        } else{
          y *= y;
          error+=y;
        }
      }
    }
  }
  error=(error) / ((patternsize.width)*patternsize.height) ;
  return error;
}
