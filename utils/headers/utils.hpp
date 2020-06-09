#ifndef _UTILS_H_
#define _UTILS_H_
#include<map>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Keyboard_Config
{
	int num_classes=2;
	int palette[6] = { 0,0,0,64,0,128 };
	double mean[3]= { 0.45734706, 0.43338275, 0.40058118 };
	double std[3]= { 0.23965294, 0.23532275, 0.2398498 };
    double scales[3]={0.5,0.75,1.0};
	cv::Size size=cv::Size(960,600);
	string model_path= "/data/nextcloud/dbc2017/files/project/keyboard1.pt";

};

struct Hand_Config
{
	int num_classes=2;
	int palette[6] = { 0,0,0,128,0,128 };
	double mean[3]= { 0.45734706, 0.43338275, 0.40058118 };
	double std[3]= { 0.23965294, 0.23532275, 0.2398498 };
	cv::Size size=cv::Size(960,600);
	string model_path= "/data/nextcloud/dbc2017/files/project/hand_seg.pt";

};


struct Keyboard_ResInfo
{
	Mat base_img;
	Mat img;
	int count_frame;
	Rect rect;
	Mat rote_M;
	Mat warp_M;
};


struct Keyboard_Info
{
	bool flag;
	Mat rote_M;
	Mat warp_M;
	Rect keyboard_rect;
	Mat rotated_img;
	Mat warp_img;
};

vector<Point> order_points(vector<Point> board_contours);
double calAngle(const int&x1, const int&y1, const int&x2, const int&y2);
cv::Rect find_rect(Mat &mask, const int &sx, const int&sy, \
			   const int &ex, const int &ey,bool &flag);
void video_to_frame(const string &video_path,const string &save_path,bool &Toflip);			   
map<string, vector<double>> Config(map<string, string>& string_config);

#endif //_UTILS_H_