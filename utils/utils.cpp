#include<opencv2/opencv.hpp>
// #include<sys/io.h>
#include<stddef.h>  // for std::size_t
#include<unistd.h>  //for access
#include <sys/stat.h>   //for mkdir
// #include <sys/types.h>
// #include<dirent.h>  

#include<map>
#include<string>
#include<iostream>
#include<vector>
#include<cmath>

using namespace std;
using namespace cv;

#define PI acos(-1)

vector<Point> order_points(vector<Point> board_contours){
	vector<int>sum_pixel,diff_pixel;
	vector<Point>keyboard_rect;
	for (std::size_t i=0; i < board_contours.size(); i++) {
		sum_pixel.push_back(board_contours[i].x + board_contours[i].y);
		diff_pixel.push_back(board_contours[i].y - board_contours[i].x);
	}

	auto ltPosition = min_element(sum_pixel.begin(), sum_pixel.end());
	auto lbPosition = max_element(diff_pixel.begin(), diff_pixel.end());
	auto rtPosition = min_element(diff_pixel.begin(), diff_pixel.end());
	auto rb_Position = max_element(sum_pixel.begin(), sum_pixel.end());

	keyboard_rect.push_back(board_contours[distance(begin(sum_pixel), ltPosition)]);
	keyboard_rect.push_back(board_contours[distance(begin(diff_pixel), lbPosition)]);
	keyboard_rect.push_back(board_contours[distance(begin(diff_pixel), rtPosition)]);
	keyboard_rect.push_back(board_contours[distance(begin(sum_pixel), rb_Position)]);
	
	return keyboard_rect;
}

double calAngle(const int&x1, const int&y1, const int&x2, const int&y2) {
	double angle = 0.0;
	int dx = x2 - x1;
	float dy = y2 - y1;
	angle = atan(dy / dx);
	return angle * 180 / PI;
}

cv::Rect find_rect(Mat &mask, const int &sx, const int&sy, \
			   const int &ex, const int &ey,bool &flag){
	int height = mask.rows, width = mask.cols;
	vector<int>loc_x, loc_y;
	for (int i = sy; i < ey; i++) {
		uchar* data = mask.ptr<uchar>(i);
		for (int j = sx; j < ex; j++) {
			if (int(data[j]) != 0) {
				loc_y.push_back(i);
				break;
			}
		}
	}
	std::sort(loc_y.begin(), loc_y.end());

	int locy_min = 0, locy_max = 0;
	//---find ymin---
	for (std::size_t i=0; i < loc_y.size(); i++) {
		int index = loc_y[i];
		uchar* data1 = mask.ptr<uchar>(index);
		int pixel_wlength=0;
		for (int j = 0; j < width; j++) {
			if (data1[j] != 0) {
				pixel_wlength++;
			}
		}
		if (pixel_wlength > 0.3 * width) {
			locy_min = index;
			break;
		}
	}
	//---find ymax---
	for (int i = loc_y.size()-1; i >= 0; i--) {
		int index = loc_y[i];
		uchar* data1 = mask.ptr<uchar>(index);
		int pixel_wlength = 0;
		for (int j = 0; j < width; j++) {
			if (data1[j] != 0) {
				pixel_wlength++;
			}
		}
		if (pixel_wlength > 0.3 * width) {
			locy_max = index;
			break;
		}
	}
	int piano_ylen = locy_max - locy_min;
	int locx_min = 0, locx_max = 0;
	//---find xmin---
	for (int i = sx; i < ex; i++) {
		int pixel_hlength = 0;
		for (int j = 0; j < height; j++) {
			if (j > locy_min&& j < locy_max) {
				if (int(mask.at<uchar>(j, i)) != 0) {
					pixel_hlength++;
				}
			}
		}
		if (pixel_hlength > 0.3 * piano_ylen) {
			locx_min = i;
			break;
		}
	}
	//---find xmax---
	for (int i = ex-1; i >= sx; i--) {
		int pixel_hlength = 0;
		for (int j = 0; j < height; j++) {
			if (j > locy_min&& j < locy_max) {
				if (int(mask.at<uchar>(j, i)) != 0) {
					pixel_hlength++;
				}
			}
		}
		if (pixel_hlength > 0.3 * piano_ylen) {
			locx_max = i;
			break;

		}
	}
	Rect board_rect(locx_min, locy_min, locx_max-locx_min, locy_max-locy_min);
	if (board_rect.height < 20) {
		flag = false;
		//return board_rect;    //--可以通过if来返回不同的值
	}
	else{
		flag=true;
	}
	return board_rect;
}


void video_to_frame(const string &video_path,const string &save_path,bool &Toflip) {

	VideoCapture capture(video_path);
	if (!capture.isOpened())
		cout << "video failed to open !" << endl;
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	long currentFrame = 0;

	// std::string save_path = "E:\\Learning\\visual_studio\\piano\\piano\\video\\video1\\";
	if (access(save_path.c_str(), 0) == -1)
		mkdir(save_path.c_str(),0775);
	long Frametostop=200;
	bool flag = true;
	Mat frame;
	while (flag) {
		if (!capture.read(frame)) {
			cout << "读取视频失败" << endl;
			return;
		}
		cout << "正在写第" << currentFrame << "帧" << endl;
		stringstream str;
		str << save_path << setw(4) << setfill('0') << right << currentFrame << ".jpg";
		cout << str.str() << endl;
		if (Toflip==true){
			flip(frame,frame,-1); //水平垂直翻转
		}
		imwrite(str.str(), frame);

		if (currentFrame >= totalFrameNumber-1)
			flag = false;

		//---frame to stop
		if (currentFrame >= Frametostop)
			flag = false;
		
		currentFrame++;
	}
	cout << "the total frame is " << totalFrameNumber << endl;
}


map<string, vector<double>> Config(map<string, string>& string_config) {
	string_config["img_paths"]= "E:\\Learning\\visual_studio\\piano\\piano\\video\\video1\\";

	double NUM_CLASSES = 2;
	double KEYBOARD_PALETTE[] = { 0, 0, 0, 64, 0, 128 };

	vector<double> num_classes, keyboard_plaette;
	num_classes.push_back(NUM_CLASSES);
	for (std::size_t i=0; i < (sizeof(KEYBOARD_PALETTE) / sizeof(KEYBOARD_PALETTE[0])); i++) {
		keyboard_plaette.push_back(KEYBOARD_PALETTE[i]);
	}

	map<string, vector<double>>number_config;
	number_config["NUM_CLASSES"] = num_classes;
	number_config["KEYBOARD_PALETTE"] = keyboard_plaette;

	return number_config;
}