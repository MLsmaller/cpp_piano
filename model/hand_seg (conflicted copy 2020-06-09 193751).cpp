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


Hand_Config hand_config;
static torch::Device device1(torch::kCPU);

using namespace cv;
using namespace std;
using namespace torch;
using namespace chrono;

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


HandSeg::HandSeg(){
    num_classes=hand_config.num_classes;
    model_path=hand_config.model_path;
    size=hand_config.size;
    memcpy(mean, hand_config.mean, sizeof(hand_config.mean));
    memcpy(std, hand_config.std, sizeof(hand_config.std));
}


void HandSeg::load_hand_model(){
    if (torch::cuda::is_available()){
        // std::cout<<"CUDA is available!"<<endl;
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


torch::Tensor HandSeg::predict(torch::Tensor image){
    
    // long int height=image.data().sizes()[2];
    // long int width=image.data().sizes()[3];
    
    torch::Tensor pred;
    auto start = system_clock::now();
        
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image.to(device1));    
    torch::Tensor prediction=(module.forward(inputs).toTensor()).to(at::kCPU);
    
    prediction=torch::squeeze(prediction,0);
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "it cost  "<< double(duration.count()) * microseconds::period::num / \
            microseconds::period::den << "  second" << endl;
    

    prediction=torch::softmax(prediction,0);
    pred=prediction.argmax(0);
    pred=pred.toType(torch::kInt);
    return pred;
}

int HandSeg::post_process(torch::Tensor mask,Rect rect){
    int h=mask.data().sizes()[0];
    int w=mask.data().sizes()[1];

    Mat mask_img(h, w, CV_32SC1,mask.data_ptr<int>());
    mask_img.convertTo(mask_img, CV_8UC1);
    Mat save_mask=mask_img.clone();
    save_mask.setTo(255,save_mask>0);  //大于0的像素设置为255
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST); //现在存的时候就是和原图一样大小
    // string save_path="/data/nextcloud/dbc2017/files/project/keyboard_images/mask_imgs/hand_";
    // stringstream str;
	// str << save_path << setw(4) << setfill('0') << right << frames_nums << ".png";
    // cout<<"save_path is "<<str.str()<<endl;
    // cv::imwrite(str.str(),save_mask);
    // frames_nums++;

    Mat roi_mask=save_mask(rect);

    int num_pixels=0;
    int h_ori=roi_mask.rows,w_ori=roi_mask.cols;
    for(int i=0;i<h_ori;i++){
        for(int j=0;j<w_ori;j++){
            if(int(roi_mask.at<uchar>(i,j))>0){
                num_pixels++;
            }
        }
    }
    // cout<<"the num_pixels is "<<num_pixels<<endl;

    return num_pixels;
}

torch::Tensor HandSeg::MatTotensor(Mat image){
    resize(image,image,size,0,0,INTER_LINEAR);
    // cout<<"the resize img size is "<<image.size()<<endl;
    torch::data::transforms::Normalize<torch::Tensor>normalize_layer =torch::data::transforms::Normalize<torch::Tensor>(\
                                                {mean[0],mean[1],mean[2]},\
                                                {std[0],std[1],std[2]});    
    Mat float_img,input;
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
    return tensor_image;
}

int HandSeg::detect_hand(Mat image,Rect rect){
    ori_size=cv::Size(image.cols,image.rows);     
    torch::Tensor tensor_image;
    torch::Tensor predictions;
    tensor_image=MatTotensor(image);
    predictions=predict(tensor_image);
    
    //---hand_seg部分post_process()函数返回值还没有完善
    int num_pixels=post_process(predictions,rect);

    return num_pixels;
}