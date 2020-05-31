#include <iostream>                                                                                                
#include "torch/script.h"
#include "torch/torch.h"
#include<opencv2/opencv.hpp>
#include<utils.hpp>
#include<seg.hpp>

#include <vector>
#include<string>
#include<cstring>
#include<memory>

using namespace std;
using namespace cv;
using namespace torch;

int main(){
    // static string model_path="/data/nextcloud/dbc2017/files/project/model.pt";
    // //  对于pytorch1.1及以下版本加载方式
    // // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
    // // assert(module != nullptr);

    Mat image=imread("/data/nextcloud/dbc2017/files/project/0146.jpg");
    cout<<"the image size is "<<image.size()<<endl;
    KeyBoard keyboard;
    keyboard.load_keyboard_model();
    keyboard.detect_keyboard(image);


    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));
    // at::Tensor output =module.forward(inputs).toTensor();
    // //若前面有module->to(at::kCUDA);将module转换为了指针则module对象可用->操作符
    // //at::Tensor output =module->forward(inputs).toTensor();
    // std::cout<<"output size is "<<output.sizes()<<endl;
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    return 0;

}