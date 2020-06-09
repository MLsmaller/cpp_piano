#include <iostream>                                                                                                
#include "torch/script.h"
#include "torch/torch.h"
#include<opencv2/opencv.hpp>
#include<utils.hpp>
#include<board_seg.hpp>
#include<hand_seg.hpp>
#include<helpers.hpp>

#include <vector>
#include<string>
#include<cstring>
#include<memory>

using namespace std;
using namespace cv;
using namespace torch;

int main(){

    // cv::String image_paths="/data/nextcloud/dbc2017/files/project/keyboard_images/";
    cv::String image_paths="/data/nextcloud/dbc2017/files/project/videos_frams/0/";
    vector<cv::String> fn;
    glob(image_paths,fn,false);

    KeyBoard keyboard;
    keyboard.load_keyboard_model();
    Keyboard_ResInfo result_Info;
    result_Info=find_base_img(keyboard,fn);
    
    Mat base_img=result_Info.base_img;
    Mat base_all_img=result_Info.img;
    Rect keyboard_rect=result_Info.rect;
    int start_frame=result_Info.count_frame;
    Mat warp_M=result_Info.warp_M;
    Mat rote_M=result_Info.rote_M;
    if(base_img.cols==0){
        return -1;
    }
    vector<Rect> total_top,total_bottom,black_boxes;
    vector<float> white_loc;
    find_key_loc(base_img,total_top,total_bottom,black_boxes,white_loc);

    return 0;

}