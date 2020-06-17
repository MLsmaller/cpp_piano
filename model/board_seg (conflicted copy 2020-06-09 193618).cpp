#include "torch/script.h"
#include "torch/torch.h"
#include<utils.hpp>  
#include<opencv2/opencv.hpp>
#include<stddef.h>

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


Keyboard_Info KeyBoard::post_process(Mat image,torch::Tensor mask){
    cout<<"now is post_process"<<endl;

	Keyboard_Info keyboard_info;
    int h=mask.data().sizes()[0];
    int w=mask.data().sizes()[1];
    Mat mask_img(h, w, CV_32SC1,mask.data_ptr<int>());
    mask_img.convertTo(mask_img, CV_8UC1);

    Mat save_mask=mask_img.clone();
    save_mask.setTo(255,save_mask>0);  //大于0的像素设置为255
    
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST); //现在存的时候就是和原图一样大小
    
	// //--save img
	// string save_path="/data/nextcloud/dbc2017/files/project/keyboard_images/mask_imgs/";
	// stringstream str;
	// str << save_path << setw(4) << setfill('0') << right << frames_nums << ".png";
    // // cout<<"save_path is "<<str.str()<<endl;
    // cv::imwrite(str.str(),save_mask);
    // frames_nums++;

	threshold(save_mask, save_mask, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(save_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());

	int max_index = 0;
	vector<Point>board_contours;
	for (std::size_t i= 0; i < contours.size(); i++) {
		if (int(contours[i].size()) > max_index) {
			max_index = contours[i].size();
			board_contours.assign(contours[i].begin(), contours[i].end());
		}
	}

	Point lt, lb, rt, rb;
	if (board_contours.size() > 500) {
		vector<Point> keyboard_rect=order_points(board_contours);
		lt = keyboard_rect[0], lb = keyboard_rect[1];
		rt = keyboard_rect[2], rb = keyboard_rect[3];
        if (abs(lt.y - rt.y) > 5 || abs(rb.y - lb.y) > 5) {
			int xb1 = lb.x, yb1 = lb.y, xb2 = rb.x, yb2 = rb.y;
			int xt1 = lt.x, yt1 = lt.y, xt2 = rt.x, yt2 = rt.y;
			Point2f center = Point2f(save_mask.cols / 2, save_mask.rows / 2);
			double angle;
			double scale = 1;
			Mat rot_mat(2, 3, CV_32FC1);
			Mat rotated_img;
			if (abs(yb1 - yb2) > abs(yt1 - yt2)) {
				angle = calAngle(xb1,yb1,xb2,yb2);
				rot_mat=getRotationMatrix2D(center, angle, scale);
                warpAffine(image, rotated_img, rot_mat, image.size());
			}
			else {
				angle = calAngle(xt1, yt1, xt2, yt2);
				rot_mat = getRotationMatrix2D(center, angle, scale);
				warpAffine(image, rotated_img, rot_mat, image.size());
			}
			keyboard_info.flag = true;
			keyboard_info.rote_M = rot_mat.clone();
			keyboard_info.rotated_img = rotated_img.clone();
		}
		else {
			int sx = min(lt.x, lb.x), ex = max(rt.x, rb.x);
			int sy = min(lt.y, rt.y), ey = max(lb.y, rb.y);
			Rect keyboard_rect1=find_rect(save_mask, sx, sy, ex, ey, keyboard_info.flag);
			keyboard_info.keyboard_rect = keyboard_rect1;
		}
	}
	else {
		keyboard_info.flag = false;
	}    
    return keyboard_info;

}

Keyboard_Info KeyBoard::post_process1(Mat image,torch::Tensor mask){
    // ori_size
	cout<<"now is post_process1"<<endl;
	Keyboard_Info keyboard_info;
    int h=mask.data().sizes()[0];
    int w=mask.data().sizes()[1];

    Mat mask_img(h, w, CV_32SC1,mask.data_ptr<int>());
    mask_img.convertTo(mask_img, CV_8UC1);
    Mat save_mask=mask_img.clone();
    save_mask.setTo(255,save_mask>0);  //大于0的像素设置为255
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST); //现在存的时候就是和原图一样大小
    // string save_path="/data/nextcloud/dbc2017/files/project/keyboard_images/mask_imgs/post1_";
	// stringstream str;
	// str << save_path << setw(4) << setfill('0') << right << (frames_nums-1) << ".png";
    // cv::imwrite(str.str(),save_mask);
    // frames_nums++;

	threshold(save_mask, save_mask, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(save_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());

	int max_index = 0;
	vector<Point>board_contours;
	for (std::size_t i= 0; i < contours.size(); i++) {
		if (int(contours[i].size()) > max_index) {
			max_index = contours[i].size();
			board_contours.assign(contours[i].begin(), contours[i].end());
		}
	}


	Point lt, lb, rt, rb;
	if (board_contours.size() > 500) {
		vector<Point> keyboard_rect=order_points(board_contours);		
		lt = keyboard_rect[0], lb = keyboard_rect[1];
		rt = keyboard_rect[2], rb = keyboard_rect[3];
        if (abs(lt.y - rt.y) > 5 || abs(rb.y - lb.y) > 5) {
			int xb1 = lb.x, yb1 = lb.y, xb2 = rb.x, yb2 = rb.y;
			int xt1 = lt.x, yt1 = lt.y, xt2 = rt.x, yt2 = rt.y;
			Mat warp_img;
			cv::Mat M;
			cv::Point2f src_points[4],dst_points[4];
			src_points[0]= cv::Point2f(lt.x, lt.y);
			src_points[1]= cv::Point2f(lb.x, lb.y);
			src_points[2]=cv::Point2f(rt.x, rt.y);
			src_points[3]=cv::Point2f(rb.x, rb.y);			
			if (abs(yb1 - yb2) > abs(yt1 - yt2)) {
				if (yb1>yb2){
					dst_points[0]=cv::Point2f(lt.x, lt.y);
					dst_points[1]=cv::Point2f(lb.x, lb.y);
					dst_points[2]=cv::Point2f(rt.x, rt.y);
					dst_points[3]=cv::Point2f(rb.x, lb.y);
				}
				else{
					dst_points[0] = cv::Point2f(lt.x, lt.y);
					dst_points[1] = cv::Point2f(lb.x, rb.y);
					dst_points[2] = cv::Point2f(rt.x, rt.y);
					dst_points[3] = cv::Point2f(rb.x, rb.y);					
				}
				M= cv::getPerspectiveTransform(src_points, dst_points);
				cv::warpPerspective(image, warp_img, M, ori_size);
			}
			else {
				if (yt1<yt2){
					dst_points[0] = cv::Point2f(lt.x, lt.y);
					dst_points[1] =cv::Point2f(lb.x, rb.y);
					dst_points[2] =cv::Point2f(rt.x, lt.y);
					dst_points[3] =cv::Point2f(rb.x, rb.y);						
				}
				else{
					dst_points[0]= cv::Point2f(lt.x, rt.y);
					dst_points[1]=cv::Point2f(lb.x, lb.y);
					dst_points[2]=cv::Point2f(rt.x, rt.y);
					dst_points[3]=cv::Point2f(rb.x, rb.y);						
				}
				M= cv::getPerspectiveTransform(src_points, dst_points);
				cv::warpPerspective(image, warp_img, M, ori_size);
			}
			keyboard_info.flag = true;
			keyboard_info.warp_M = M.clone();
			keyboard_info.warp_img = warp_img.clone();
		}
		else {
			int sx = min(lt.x, lb.x), ex = max(rt.x, rb.x);
			int sy = min(lt.y, rt.y), ey = max(lb.y, rb.y);
			Rect keyboard_rect1=find_rect(save_mask, sx, sy, ex, ey, keyboard_info.flag);
			keyboard_info.keyboard_rect = keyboard_rect1;
		}
	}
	else {
		keyboard_info.flag = false;
	}    
    return keyboard_info;
}

Keyboard_Info KeyBoard::post_process2(Mat image,torch::Tensor mask){
    cout<<"now is post_process2"<<endl;
	Keyboard_Info keyboard_info;
    int h=mask.data().sizes()[0];
    int w=mask.data().sizes()[1];

    Mat mask_img(h, w, CV_32SC1,mask.data_ptr<int>());
    mask_img.convertTo(mask_img, CV_8UC1);
    Mat save_mask=mask_img.clone();
    save_mask.setTo(255,save_mask>0);  //大于0的像素设置为255
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST); //现在存的时候就是和原图一样大小
    // string save_path="/data/nextcloud/dbc2017/files/project/keyboard_images/mask_imgs/post2_";
	// stringstream str;
	// str << save_path << setw(4) << setfill('0') << right << (frames_nums-1) << ".png";
    // cv::imwrite(str.str(),save_mask);
    // frames_nums++;


	threshold(save_mask, save_mask, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(save_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());

	int max_index = 0;
	vector<Point>board_contours;
	for (std::size_t i= 0; i < contours.size(); i++) {
		if (int(contours[i].size()) > max_index) {
			max_index = contours[i].size();
			board_contours.assign(contours[i].begin(), contours[i].end());
		}
	}

	Point lt, lb, rt, rb;
	if (board_contours.size() > 500) {
		vector<Point> keyboard_rect;
		keyboard_rect=order_points(board_contours);		
		lt = keyboard_rect[0], lb = keyboard_rect[1];
		rt = keyboard_rect[2], rb = keyboard_rect[3];
		int sx = min(lt.x, lb.x), ex = max(rt.x, rb.x);
		int sy = min(lt.y, rt.y), ey = max(lb.y, rb.y);
		Rect keyboard_rect1=find_rect(save_mask, sx, sy, ex, ey, keyboard_info.flag);
		keyboard_info.keyboard_rect = keyboard_rect1;
	}
	else{
		keyboard_info.flag=false;
	}
	return keyboard_info;
}


torch::Tensor KeyBoard::multi_scale_predict(torch::Tensor image,const int &multi_scale){
    
    long int height=image.data().sizes()[2];
    long int width=image.data().sizes()[3];

    torch::nn::Upsample sample_layer=nn::Upsample(nn::UpsampleOptions()\
                              .size(std::vector<long int>({height,width}))\
                              .mode(torch::kBilinear).align_corners(false));
    
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
                                .mode(torch::kBilinear).align_corners(false));

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
        cout<<"\n";
		// cout<<total_predictions.data()<<endl;
        total_predictions=torch::softmax(total_predictions,0);
        pred=total_predictions.argmax(0);
        pred=pred.toType(torch::kInt);
    }

    return pred;
}

torch::Tensor KeyBoard::MatTotensor(Mat image){
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

Keyboard_Info KeyBoard::detect_keyboard(Mat image){
    ori_size=cv::Size(image.cols,image.rows);     
    torch::Tensor tensor_image;
    torch::Tensor predictions;
    tensor_image=MatTotensor(image);
	cout<<"now is the first predict"<<endl;
    predictions=multi_scale_predict(tensor_image,multi_scale);
    
    Mat float_img;
    image.convertTo(float_img,CV_32FC3);
    Keyboard_Info keyboard_info;
    keyboard_info=post_process(float_img,predictions);
	
    if (keyboard_info.flag==false){
		return keyboard_info;
	}
	if (keyboard_info.keyboard_rect.width!=0){
		return keyboard_info;
	}

	Mat rotated_img = keyboard_info.rotated_img.clone();
    // imwrite("../rotated.jpg",rotated_img);
    tensor_image =MatTotensor(rotated_img);

    predictions=multi_scale_predict(tensor_image,multi_scale);    
    
	// cout<<"\n"<<endl;
	
    rotated_img.convertTo(float_img,CV_32FC3);
	Keyboard_Info keyboard_info1;
    keyboard_info1=post_process1(float_img,predictions);

	keyboard_info1.rotated_img=rotated_img.clone();
	keyboard_info1.rote_M=keyboard_info.rote_M.clone();

    if (keyboard_info1.flag==false){
		return keyboard_info1;
	}	
	if (keyboard_info1.keyboard_rect.width!=0){
		return keyboard_info1;
	}
	Mat warp_img=keyboard_info1.warp_img.clone();
    tensor_image =MatTotensor(warp_img);
    predictions=multi_scale_predict(tensor_image,multi_scale);    
    
    warp_img.convertTo(float_img,CV_32FC3);
	Keyboard_Info keyboard_info2;
    keyboard_info2=post_process2(float_img,predictions);

    if (keyboard_info2.flag==false){
		return keyboard_info2;
	}	
	keyboard_info2.warp_M=keyboard_info1.warp_img.clone();
	keyboard_info2.rote_M=keyboard_info.rote_M.clone();
	keyboard_info2.warp_img=warp_img.clone();
	keyboard_info2.rotated_img=rotated_img.clone();
    return keyboard_info2;
}