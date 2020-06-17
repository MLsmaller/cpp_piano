#ifndef _HELPERS_H_
#define _HELPERS_H_

#include<opencv2/opencv.hpp>
#include<utils.hpp>
#include<board_seg.hpp>


#include<vector>
#include<string>

using namespace cv;

vector<Rect> find_black_boxes1(Mat &ori_img,vector<int> &black_loc);
float get_light(vector<cv::String> fn);
Keyboard_ResInfo find_base_img(KeyBoard keyboard,vector<cv::String> fn);
void find_key_loc(Mat &base_img);
void find_black_boxes(Mat &ori_img);
Mat remove_region(Mat &img);
vector<Rect> find_black_keys(Mat &base_img);
bool rect_compare(Rect i, Rect j);
vector<Rect> find_black_boxes(Mat &ori_img,vector<int> &black_loc);
vector<float> find_white_loc(vector<int> &black_loc,vector<Rect> &black_boxes,const int &width);
int near_white(const float &white_loc,const vector<Rect> &black_boxes);
vector<int> white_black_dict();
void find_key_loc(Mat &base_img,vector<Rect> &total_top,vector<Rect> &total_bottom, \
                  vector<Rect> &black_boxes,vector<float> &white_loc);


#endif // _HELPERS_H_