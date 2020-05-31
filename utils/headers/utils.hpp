#ifndef _UTILS_H_
#define _UTILS_H_
#include<map>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>

using namespace std;

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

map<string, vector<double>> Config(map<string, string>& string_config);

#endif