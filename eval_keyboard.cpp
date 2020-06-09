#include "torch/script.h"  //--在main函数中要包含这两个头文件，不然会报一些奇怪的错误
#include "torch/torch.h"

#include <iostream>                                                                                                
#include<opencv2/opencv.hpp>
#include<utils.hpp>
#include<helpers.hpp>
#include<dirent.h>

#include <vector>
#include<string>
#include<cstring>
#include<memory>

using namespace std;
using namespace cv;


vector<string> get_path(string rootdirPath){
    DIR * dir;
	//---定义指针的时候最好要初始化，如果不初始化在函数return时会报Segmentation fault的错误
	//--如果在定义时没有初始化，在给指针对象时加上new来申请分配空间，记得最后delete删除指针
    struct dirent * ptr=NULL;   //结构体指针
	vector<string>video_path;
    string x,dirPath;
    dir = opendir((char *)rootdirPath.c_str()); //打开一个目录
    // ptr=new readdir(dir);
	while((ptr=readdir(dir)) != NULL) //循环读取目录数据
    {
        x=ptr->d_name;
		// string::size_type idx=x.find_first_of(".");
		string::size_type idx=x.find("."); //有.和..两个目录
		if (int(idx)!=0){
			//---判断路径是不是以/结尾
			if(rootdirPath.rfind("/")==(rootdirPath.length()-1))
        		dirPath = rootdirPath + x;
			else
				dirPath = rootdirPath + "/"+x;
			// cout<<"dir name "<< dirPath.c_str()<<endl;
        	video_path.push_back(dirPath.c_str()); //存储到数组
		}
    }
	delete ptr;
    closedir(dir);
	return video_path;
}

int main(int argc,char **argv) {
	string img_path = "/data/nextcloud/dbc2017/files/project/keyboard_images/rect/rect_mask0000.png";
	Mat img = imread(img_path);
	// find_key_loc(img);

    // static string model_path="/data/nextcloud/dbc2017/files/project/model.pt";
    // //  对于pytorch1.1及以下版本加载方式
    // // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);
    // // assert(module != nullptr);

	//===--get video paths--====
    string rootdirPath = "/data/nextcloud/dbc2017/files/project/videos_frams/";
	vector<string> video_dirs=get_path(rootdirPath);
	vector<string>video_paths;
	for (std::size_t i=0;i<video_dirs.size();i++){
		vector<string> paths=get_path(video_dirs[i]);
		// video_paths.insert(video_paths.end(),paths.begin(),paths.end());
		for(std::size_t j=0;j<paths.size();j++){
			if(paths[j].find(".wmv")<2000){
				video_paths.push_back(paths[j]);
			}
		}
	}

	for(std::size_t i=0;i<video_paths.size();i++){
		cout<<video_paths[i]<<endl;
		int pos =video_paths[i].rfind("/");
		string save_path=video_paths[i].substr(0,pos+1);
		// cout<<save_path<<endl;
		bool flag=true;
		video_to_frame(video_paths[i],save_path,flag);
	}


	return 0;
}