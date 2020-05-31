#ifndef _SEG_H_
#define _SEG_H_

#include "torch/script.h"
#include "torch/torch.h"
#include<opencv2/opencv.hpp>

using namespace cv;

class KeyBoard{
    public:
        KeyBoard();
        void load_keyboard_model();
        void detect_keyboard(Mat image);
        
    private:
        double mean[3], std[3],scales[3];
        double palette[6];
        int num_classes;
        cv::Size size;
        string model_path;
        int multi_scale=0;
        torch::jit::script::Module module;  //如果在类的成员函数里面定义不了可以在外面进行全局定义static
        // multi_scale=1-> multi scale  else single scale
        torch::Tensor multi_scale_predict(torch::Tensor image,const int &multi_scale);
        void post_process(Mat image,torch::Tensor mask);
};

#endif _SEG_H_