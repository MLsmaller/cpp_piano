

#ifndef _HAND_SEG_H_
#define _HAND_SEG_H_

#include "torch/script.h"
#include "torch/torch.h"
#include<opencv2/opencv.hpp>
#include<utils.hpp>

using namespace cv;

class HandSeg{
    public:
        HandSeg();
        void load_hand_model();
        int detect_hand(Mat image,Rect rect);
    private:
        int frames_nums=0;
        double mean[3], std[3];
        int num_classes;
        cv::Size size;
        cv::Size ori_size;
        string model_path;
        torch::jit::script::Module module;  //如果在类的成员函数里面定义不了可以在外面进行全局定义static
        // multi_scale=1-> multi scale  else single scale
        torch::Tensor predict(torch::Tensor image);
        torch::Tensor MatTotensor(Mat image);
        int post_process(torch::Tensor mask,Rect rect);
};


#endif //_HAND_SEG_H_

