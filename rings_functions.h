#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <time.h>
#include <vector>
#include <math.h>
#include <ctime>
#include "integral_threshold.h"


using namespace std;
using namespace cv;

float n_x = 10, n_y = 8, dsty;
vector<vector<int> > dd(n_y, vector<int>(n_x));

void trace_line(Mat &frame, vector<Point2f> arrange)
{
    for (size_t i = 0; i < 4; i++)
        line(frame, arrange[i], arrange[i+1], Scalar(255,0,0), 2, 8, 0);
    line(frame, arrange[4], arrange[5], Scalar(180,180,0), 2, 8, 0);
    for (size_t i = 5; i < 9; i++)
        line(frame, arrange[i], arrange[i+1], Scalar(0,255,0), 2, 8, 0);
    line(frame, arrange[9], arrange[10], Scalar(0,255,180), 2, 8, 0);
    for (size_t i = 10; i < 14; i++)
        line(frame, arrange[i], arrange[i+1], Scalar(0,255,255), 2, 8, 0);
    line(frame, arrange[14], arrange[15], Scalar(180,0,255), 2, 8, 0);
    for (size_t i = 15; i < 19; i++)
        line(frame, arrange[i], arrange[i+1], Scalar(255,0,255), 2, 8, 0);
}

float euclideanDist(Point p, Point q)
{
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int get_opposite(vector<Point2f> points, Point2f p)
{
    float mx = 0;
    int op_ind;
    for (size_t i = 0; i < points.size(); i++)
    {
        if(mx < euclideanDist(p, points[i]))
        {
            mx = euclideanDist(p, points[i]);
            op_ind = i;
        }
    }
    return op_ind;
}

Point2f get_vector(Point2f p, Point2f q)
{
    Point2f axis;  
    float dis = euclideanDist(p, q);
    axis.x = (p.x - q.x)/dis;
    axis.y = (p.y - q.y)/dis;
    return axis;
}

void first_line(vector<Point2f> input, vector<Point2f> &first_row, int &k, Point2f axis_x, float dis_x)
{
    bool flag_end = 0;
    while (!flag_end)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            flag_end = 1;
            if (euclideanDist(input[k] + dis_x*axis_x, input[i]) < dis_x/4.0)
            {
                flag_end = 0;
                dis_x = euclideanDist(input[i], input[k]);
                k = i;
                first_row.push_back(input[k]);
                break;
            }
        }
    }
}

void axis_correction(vector<Point2f> &first_row, Point2f &axis_x, Point2f &axis_y, vector<int> &corners, bool &next_row)
{
    next_row = 1;
    int sw;
    if (first_row.size() == 5) //si es eje x
    {
        if(axis_x.x < 0) // invertido
        {
            axis_x.x = -axis_x.x;
            axis_x.y = -axis_x.y;
            reverse(first_row.begin(),first_row.end());
            // primer elemento de la linea
            sw = corners[0];
            corners[0] = corners[1];
            corners[1] = sw;
        }
    }
    else
    {
        next_row = 0;
        Point2f ax = axis_x;
        axis_x = axis_y;
        axis_y = ax;
        corners[2] = corners[1];
        if(axis_y.x > 0)
        {
            axis_x.x = -axis_x.x;
            axis_x.y = -axis_x.y;
            sw = corners[0];
            corners[0] = corners[2];
            corners[1] = sw;
            corners[3] = corners[1];
        }
    }
}

void fill_arrange(vector<Point2f> input, vector<Point2f> &output, bool next_row,vector<int> &corners, Point2f axis_x,float dis_x, Point2f axis_y, float dis_y)
{
    bool flag_end = 0, fill = !next_row;
    int f_row = corners[0];
    int actual = corners[0];

    while(!flag_end)
    {
        if (next_row)
        {
            next_row = 0;
            for (size_t j = 0; j < input.size(); j++)
            {
                flag_end = 1;
                if (euclideanDist(input[f_row] + dis_y*axis_y, input[j]) < dis_y/4.0)
                {
                    dis_y = euclideanDist(input[f_row], input[j]);
                    flag_end = 0;
                    f_row = j;
                    actual = j;
                    corners[2] = j;
                    output.push_back(input[f_row]);
                    break;
                }
            }
        }
        for (size_t j = 0; j < input.size(); j++) //eje x
        {
            next_row = 1;
            if (euclideanDist(input[actual] + dis_x*axis_x, input[j]) < dis_x/4.0)
            {
                dis_x = euclideanDist(input[actual], input[j]);
                next_row = 0;
                actual = j;
                output.push_back(input[actual]);
                break;
            }
        }
        if(fill)
        {
            fill = 0;
            corners[2] = actual;
        }
    }
}

void matching_normal(vector<Point2f> &arrange, vector<Point2f> points)
{
    int min_frs, new_d_frs, indx;
    for (size_t i = 0; i < arrange.size(); i++)
    {
        min_frs = 1000;
        for (size_t j = 0; j < points.size(); j++)
        {
            new_d_frs = euclideanDist(arrange[i], points[j]);
            if(min_frs > new_d_frs)
            {
                min_frs = new_d_frs;
                indx = j;//nuevo punto
            }
        }
        arrange[i] = points[indx];
        points.erase (points.begin()+indx);
    }
}

void first_function(vector<Point2f> points, vector<Point2f> &arrange)
{
    vector<int> corners(4);
    bool nr;
    Point2f axis_x, axis_y;
    float dis_x, dis_y;
    arrange.clear();
    axis_x = get_vector(points[1], points[0]);
    dis_x = euclideanDist(points[0], points[1]);
    int op_ind = get_opposite(points, points[0]);
    arrange.push_back(points[0]);
    arrange.push_back(points[1]);
    corners[0] = 0;
    corners[1] = 1;
    corners[3] = op_ind;
    first_line(points, arrange, corners[1], axis_x , dis_x);
    axis_y = get_vector(points[op_ind], points[corners[1]]);
    dis_y = dis_x;
    axis_correction(arrange, axis_x, axis_y, corners, nr);
    fill_arrange(points, arrange , nr, corners, axis_x, dis_x, axis_y, dis_y);
}

vector<Point2f> get_limits(vector<Point2f> arrange)
{
    vector<Point2f> limits;
    Point2f min_c, max_c;
    max_c.x = max(max(arrange[0].x, arrange[4].x), max(arrange[15].x, arrange[19].x));
    max_c.y = max(max(arrange[0].y, arrange[4].y), max(arrange[15].y, arrange[19].y));
    min_c.x = min(min(arrange[0].x, arrange[4].x), min(arrange[15].x, arrange[19].x));
    min_c.y = min(min(arrange[0].y, arrange[4].y), min(arrange[15].y, arrange[19].y));
    limits.push_back(min_c);
    limits.push_back(max_c);
    return limits;
}

void limits_density(vector<vector<Point2f> > &int_points, vector<Point2f> arrange, vector<Point2f> &crs)
{
    vector<Point2f> lt_arrange = get_limits(arrange);
    crs[1].x = max(lt_arrange[1].x, crs[1].x);
    crs[1].y = max(lt_arrange[1].y, crs[1].y);
    crs[0].x = min(lt_arrange[0].x, crs[0].x);
    crs[0].y = min(lt_arrange[0].y, crs[0].y);
    int_points.push_back(arrange);
}

void update_density(Mat &frame, vector<Point2f> points, string name)
{
    for (size_t j = 0; j < points.size(); j++)
    {
        circle(frame, points[j], 2, Scalar(0, 0, 0), -1);
    //imshow( name, frame );
    }
}

void see_density(Mat &frame, vector<vector<Point2f> > vector_points, string name)
{
    for (size_t i = 0; i < vector_points.size(); i++)
        update_density(frame, vector_points[i], name);
    name.append(".jpg");
    imwrite(name, frame);
}

void plot_density()
{
    for (size_t i = 0; i < dd.size(); i++)
    {
        std::cout << '\n';
        for (size_t j = 0; j < dd[i].size(); j++)
            std::cout << " " << dd[i][j];
    }
    std::cout << '\n';
}

bool dd_evaluation(vector<Point2f> lts)
{
    bool flag = 1;
    float eva = 0, total = (lts[1].x-lts[0].x+1)*(lts[1].y-lts[0].y+1);

    for (size_t i = lts[0].x; i <= lts[1].x; i++)
        for (size_t j = lts[0].y; j <= lts[1].y; j++)
            if (dd[j-1][i-1] > dsty)
                eva++;
    if (eva/total > 0.45)
        flag = 0;
    return flag;
}

void get_dd_limits(vector<Point2f> arrange, vector<Point2f> crs, vector<Point2f> &lts, float p_x, float p_y)
{
    bool goi = 1, goj = 1;
    vector<Point2f> lt_arrange = get_limits(arrange);

    // Eje X
    for (size_t i = 1; i <= n_x; i++)
    {
        if(goi && lt_arrange[0].x < crs[0].x + i*p_x)
        {
            goi = 0;
            lts[0].x = i;
        }
        if(lt_arrange[1].x < crs[0].x + i*p_x)
        {
            lts[1].x = i;
            break;
        }
    }

    // Eje Y
    for (size_t j = 1; j <= n_y; j++)
    {
        if(goj && lt_arrange[0].y < crs[0].y + j*p_y)
        {
            goj = 0;
            lts[0].y = j;
        }
        if(lt_arrange[1].y < crs[0].y + j*p_y)
        {
            lts[1].y = j;
            break;
        }
    }
}

bool affection(vector<Point2f> arrange, vector<Point2f> crs, vector<vector<Point2f> > &output)
{
    vector<Point2f> lts;
    lts.push_back(Point2f(0, 0)); //min
    lts.push_back(Point2f(0, 0)); //max

    float p_x = (crs[1].x - crs[0].x)/n_x;
    float p_y = (crs[1].y - crs[0].y)/n_y;

    get_dd_limits(arrange, crs, lts, p_x, p_y);
    bool flag = dd_evaluation(lts);

    if(flag)
        for (size_t k = 0; k < arrange.size(); k++)
            for (size_t i = lts[0].x; i <= lts[1].x; i++)
                if((i-1)*p_x < arrange[k].x && arrange[k].x < i*p_x)
                    for (size_t j = lts[0].y; j <= lt3d calibration s[1].y; j++)
                        if((j-1)*p_y < arrange[k].y && arrange[k].y < j*p_y)
                            dd[j-1][i-1]++;
    return flag;
}

void normalize_density_nuf(vector<Point2f> crs, vector<vector<Point2f> > input,vector<vector<Point2f> > &output, vector<Mat> int_frames,vector<Mat> &fin_frames, int nuf)
{
    dsty = nuf*20/(n_x*n_y);
    vector<int> indx;
    bool flag;
    for (size_t i = 0; i < input.size(); i++)
          indx.push_back(i);
    srand ( unsigned ( time(0) ) ); //random_shuffle
    random_shuffle ( indx.begin(), indx.end() );

    for (size_t i = 0; i < input.size(); i++)
    {
        flag = affection(input[indx[i]], crs, output);
        if(flag)
        {
            output.push_back(input[indx[i]]);
            fin_frames.push_back(int_frames[indx[i]]);
        }
        if(output.size() == nuf)  break;
    }
}

void get_random_samples(vector<vector<Point2f> > input, vector<vector<Point2f> > &output,vector<Mat> &int_frames, vector<Mat> &fin_frames, int nuf)
{
    vector<int> indx;
    for (size_t i = 0; i < input.size(); i++)
        indx.push_back(i);

    srand ( unsigned ( time(0) ) ); //random_shuffle
    random_shuffle ( indx.begin(), indx.end() );

    int sv = int(input.size()/nuf)-1;
    int k = 0;
    while(output.size() < nuf)
    {
        output.push_back(input[indx[k*sv]]);
        fin_frames.push_back(int_frames[indx[k*sv]]);
        k++;
    }
} 

vector<Point2f> get_keypoints(Mat frame,int fronto=0)
{
	int max_size=256;
	if(fronto==1)
		max_size=360;
	vector<Point2f> center;
	Mat edges,gray,gaussian;
    	vector<vector<Point>>contours;
	Point2f center1,center2,mass_center;
	RotatedRect minRect1,minRect2;
	Point2f rect_points1[4],rect_points2[4];
	vector<Vec4i> hierarchy;
	int aux,aux2;
	cvtColor(frame,gray,cv::COLOR_RGB2GRAY);
	GaussianBlur(gray,gaussian,Size(3,3),0,0);
	edges=integral_threshold(gaussian,0.85);
	if(fronto==1)
		imwrite("edges.jpg",frame);

	hierarchy.clear();
	findContours(edges,contours,hierarchy,CV_RETR_TREE,CHAIN_APPROX_NONE,Point(0,0));
	//hierarchy [Next, Previous, First_Child, Parent]
	
	if(contours.size()<20*2)
		return center;

	if(fronto==1)
	{
		RotatedRect minEllipse;
		for(int i=0;i<(int)contours.size();++i)
			if(contours[i].size()>5)
			{
				minEllipse=fitEllipse(Mat(contours[i]));
				ellipse(frame,minEllipse,Scalar(0,255,0),1,CV_AA);
			}
	}
		
	if(fronto==1)
		imwrite("detectado.jpg",frame);

	for(int i=0;i<(int)contours.size();++i)
		if(contours[i].size()>max_size)
		{
			aux=hierarchy[i][2];
			while(aux!=-1)
			{
				hierarchy[aux][3]=hierarchy[i][3];
				aux=hierarchy[aux][0];
			}
			if(hierarchy[hierarchy[i][3]][2]==i)
				hierarchy[hierarchy[i][3]][2]=hierarchy[i][0];
			hierarchy[i][2]=-1;
			hierarchy[i][3]=-1;
		}
		else
			if(contours[i].size()<16)
			{
				hierarchy[i][2]=-1;
				if(hierarchy[i][3]!=-1)
				{
					if(hierarchy[hierarchy[i][3]][2]==i)
						hierarchy[hierarchy[i][3]][2]=hierarchy[i][0];
					hierarchy[i][3]=-1;
				}
			}

	for(int i=0;i<(int)contours.size();++i)
		if(hierarchy[i][3]!=-1 && (hierarchy[i][0]!=-1 || hierarchy[i][1]!=-1))
		{
			hierarchy[hierarchy[i][3]][2]=-1;
			hierarchy[hierarchy[i][3]][3]=-1;
			hierarchy[i][2]=-1;
			hierarchy[i][3]=-1;
		}
	for(int i=0;i<(int)contours.size();++i)
		if(hierarchy[i][2]!=-1)
		{
			minRect1=minAreaRect(Mat(contours[i]));
			minRect2=minAreaRect(Mat(contours[hierarchy[i][2]]));
			minRect1.points(rect_points1);
			minRect2.points(rect_points2);
			center1=Point((rect_points1[0].x+rect_points1[2].x)/2,(rect_points1[0].y+rect_points1[2].y)/2);
			center2=Point((rect_points2[0].x+rect_points2[2].x)/2,(rect_points2[0].y+rect_points2[2].y)/2);
			if(sqrt(pow((center1.x-center2.x),2)+pow((center1.y-center2.y),2))<4)
				center.push_back(Point((center1.x+center2.x)/2,(center1.y+center2.y)/2));
		}
	/*if(center.size()<NUM_RINGS) //early stop
		return center;*/
	while(center.size()>20) //&& center.size()!=0)
	{
		mass_center=Point(0,0);
		for(int i=0;i<(int)center.size();++i)
			mass_center+=center[i];
		mass_center.x/=center.size();
		mass_center.y/=center.size();
		aux=0;
		aux2=sqrt(pow((mass_center.x-center[0].x),2)+pow((mass_center.y-center[0].y),2));
		for(int i=1;i<(int)center.size();++i)
			if(sqrt(pow((mass_center.x-center[i].x),2)+pow((mass_center.y-center[i].y),2))>aux2)
			{
				aux=i;
				aux2=sqrt(pow((mass_center.x-center[i].x),2)+pow((mass_center.y-center[i].y),2));
			}
		center.erase(center.begin()+aux);
	}
	return center;
}

void detect_rings(Mat &frame, vector<vector<Point2f> > &int_points, bool &first_flag,vector<Point2f> &corners, vector<Mat> &int_frames, vector<Point2f> &arrange)
{
	vector<Point2f> points=get_keypoints(frame);
	if (points.size() == 20)
	{
		if(first_flag)
        {
            first_flag = 0;
            first_function(points, arrange);
            if(arrange.size() != 20)
            {
                first_flag = 1;
            }
            else
            {
                limits_density(int_points, arrange, corners);
                int_frames.push_back(frame.clone());
                trace_line(frame, arrange);
                //update_density(bg_density, arrange, "Real Time");
            }
        }
        else
        {
            matching_normal(arrange, points);
            if(get_opposite(points, points[0]) == 19 && arrange.size() == 20)
            {
                limits_density(int_points, arrange, corners);
                int_frames.push_back(frame.clone());
                trace_line(frame, arrange);
                //update_density(bg_density, arrange, "Real Time");
            }
            else
            {
                first_flag = 1;
            }
        }
    }
    else
    {
    	first_flag = 1;
    }
}

double calibrate_function(Size patternsize, Size imageSize, float rm, Mat &cameraMatrix, Mat &distCoeffs, vector<vector<Point2f> > fin_points)
{
    vector<Point3f> vectorPoints;
    vector<vector<Point3f> > objectPoints;
    vector<Mat> rvecs, tvecs;


    for( int i = 0; i < patternsize.height; ++i )
        for( int j = 0; j < patternsize.width; ++j )
            vectorPoints.push_back(Point3f(float(j)*rm, float(i)*rm, 0));

    for (int k = 0; k < fin_points.size(); k++)
        objectPoints.push_back(vectorPoints);
    
    //std::cout << objectPoints.size() << '\n';
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    distCoeffs = Mat::zeros(8, 1, CV_64F);
    double rms = calibrateCamera(objectPoints, fin_points, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "error = "<< rms << '\n';
    std::cout << "cameraMatrix = "<< cameraMatrix << '\n';
    std::cout << "distCoeffs = "<< distCoeffs << '\n';
    return rms;
}
