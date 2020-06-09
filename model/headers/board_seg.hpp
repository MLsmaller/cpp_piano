#ifndef _BOARD_SEG_H_
#define _BOARD_SEG_H_

#include "torch/script.h"
#include "torch/torch.h"
#include<opencv2/opencv.hpp>
#include<utils.hpp>

using namespace cv;

class KeyBoard{
    public:
        KeyBoard();
        void load_keyboard_model();
        Keyboard_Info detect_keyboard(Mat image);
    private:
        int frames_nums=0;
        double mean[3], std[3],scales[3];
        double palette[6];
        int num_classes;
        cv::Size size;
        cv::Size ori_size;
        string model_path;
        int multi_scale=0;
        torch::jit::script::Module module;  //如果在类的成员函数里面定义不了可以在外面进行全局定义static
        // multi_scale=1-> multi scale  else single scale
        torch::Tensor multi_scale_predict(torch::Tensor image,const int &multi_scale);
        Keyboard_Info post_process(Mat image,torch::Tensor mask);
        Keyboard_Info post_process1(Mat image,torch::Tensor mask);
		Keyboard_Info post_process2(Mat image,torch::Tensor mask);
        torch::Tensor MatTotensor(Mat image);
};


#endif //_BOARD_SEG_H_