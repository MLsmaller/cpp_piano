#include "torch/script.h"
#include "torch/torch.h"
#include<utils.hpp>
#include<opencv2/opencv.hpp>

#include<iostream>
#include<cstring>
#include<string>
#include <memory>
#include<fstream>
#include <chrono>

Keyboard_Config keyboard_config;
static torch::Device device1(torch::kCPU);

using namespace cv;
using namespace std;
using namespace torch;
using namespace chrono;

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


KeyBoard::KeyBoard(){
    num_classes=keyboard_config.num_classes;
    model_path=keyboard_config.model_path;
    size=keyboard_config.size;
    memcpy(palette, keyboard_config.palette, sizeof(keyboard_config.palette));    
    memcpy(mean, keyboard_config.mean, sizeof(keyboard_config.mean));
    memcpy(std, keyboard_config.std, sizeof(keyboard_config.std));
    memcpy(scales, keyboard_config.scales, sizeof(keyboard_config.scales));
}


void KeyBoard::load_keyboard_model(){
    if (torch::cuda::is_available()){
        std::cout<<"CUDA is available!"<<endl;
        device1=torch::Device(torch::kCUDA,0);
    }       

    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit(1);
    }    
    module.to(device1);
}

void KeyBoard::post_process(Mat image,torch::Tensor mask){
    int w,h;
    w=image.cols;
    h=image.rows;

    Mat mask_img(h, w, CV_32SC1,mask.data<int>());
    mask_img.convertTo(mask_img, CV_8UC1);
    Mat save_mask=mask_img.clone();
    save_mask.setTo(255,save_mask>0);  //大于0的像素设置为255
    cout<<save_mask.size()<<endl;
    cv::imwrite("mask.png",save_mask);

    // save_mask.convertTo(save_mask, CV_8FC1);
    threshold(save_mask,save_mask,150,255,THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
    
    //目前来看好像有了这个mask分割图像之后可以直接读取进来进行后面的操作，暂时不用gpu
	findContours(save_mask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE,Point());

}


torch::Tensor KeyBoard::multi_scale_predict(torch::Tensor image,const int &multi_scale){
    
    long int height=image.data().sizes()[2];
    long int width=image.data().sizes()[3];


    torch::nn::Upsample sample_layer=nn::Upsample(nn::UpsampleOptions()\
                              .size(std::vector<long int>({height,width}))\
                              .mode(torch::kBilinear).align_corners(true));
    
    torch::Tensor total_predictions=torch::zeros({num_classes,height,width});    
    
    int length_scale;
    if (multi_scale==1){
        length_scale=sizeof(scales)/sizeof(scales[0]);
    }
    else{
        length_scale=1;
        scales[0]=1.0;
    }
    
    torch::Tensor pred;
    for (int i=0;i<length_scale;i++){
        auto start = system_clock::now();
        double scale=scales[i];
        // 这个sample_scale其实是多余的，网络返回的就是原图的大小
        torch::nn::Upsample sample_scale=nn::Upsample(nn::UpsampleOptions()\
                                .scale_factor(std::vector<double>({scale,scale}))\
                                .mode(torch::kBilinear));

        torch::Tensor scale_image=sample_scale->forward(image);
        //--现在module不是指针了（老版本才是）
        // torch::Tensor result=module->forward(scale_image).toTensor();
            
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(scale_image.to(device1));    
        torch::Tensor prediction=(sample_layer->forward(module.forward(\
                                                inputs).toTensor())).to(at::kCPU);
        
        total_predictions+=torch::squeeze(prediction,0);
        total_predictions/=float(length_scale);

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "it cost  "<< double(duration.count()) * microseconds::period::num / \
                microseconds::period::den << "  second" << endl;
        
        // cout<<total_predictions.data()<<endl;
        total_predictions=torch::softmax(total_predictions,0);
        pred=total_predictions.argmax(0);
        pred=pred.toType(torch::kInt);
    }

    return pred;
}


void KeyBoard::detect_keyboard(Mat image){

    resize(image,image,size,0,0,INTER_LINEAR);
    cout<<"the resize img size is "<<image.size()<<endl;
    torch::data::transforms::Normalize<torch::Tensor>normalize_layer =torch::data::transforms::Normalize<torch::Tensor>(\
                                                {mean[0],mean[1],mean[2]},\
                                                {std[0],std[1],std[2]});     
    
    cv::Mat float_img,input;
	image.convertTo(float_img,CV_32FC3);
    cv::cvtColor(float_img,input,cv::COLOR_BGR2RGB);

    torch::Tensor tensor_image=torch::from_blob(input.data,{1,input.rows, \
                                                input.cols,3}, torch::kFloat);    
    tensor_image=tensor_image.permute({0,3,1,2}); //  (b,c,h,w)
    tensor_image=tensor_image.toType(torch::kFloat);
    // torch::Tensor test_img=torch::squeeze(tensor_image,0);

    tensor_image=tensor_image.div(255);
    tensor_image=normalize_layer(tensor_image);
    tensor_image=tensor_image.to(device1);
    torch::Tensor predictions=multi_scale_predict(tensor_image,multi_scale);
    
    post_process(float_img,predictions);

}