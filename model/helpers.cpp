#include <iostream>                                                                                                
#include<opencv2/opencv.hpp>
#include<assert.h>
// #include<utils.hpp>   //--不要重复include，因为board_seg中已经include了utils.hpp,不然会报奇怪的错误
#include<board_seg.hpp>
#include<hand_seg.hpp>

#include <vector>
#include<string>
#include<algorithm>

using namespace std;
using namespace cv;
vector<Rect> find_black_boxes1(Mat &ori_img,vector<int> &black_loc);
vector<Rect> find_black_boxes(Mat &ori_img,vector<int> &black_loc);

float get_light(vector<cv::String> fn){
    vector<float>ave_light;
    for(std::size_t i=0;i<fn.size();i++){
        Mat img=imread(fn[i],0);  //--for gray
        //--mean()返回的是通道的均值,现在是灰度图像，就一个通道，因此取.val[0]，下标为0
        Scalar scalar=mean(img);
        ave_light.push_back(scalar.val[0]);
    }
	float sum = std::accumulate(std::begin(ave_light), std::end(ave_light), 0.0);
	float mean =  sum / ave_light.size(); //均值
    cout<<"the mean light is "<<mean<<endl;
    return mean;

}

//---有些视频是在钢琴的最后才出现没有手的情况，对于这种如果从头开始遍历找背景图太花时间，应该换种策略
Keyboard_ResInfo find_base_img(KeyBoard keyboard,vector<cv::String> fn){
    Keyboard_ResInfo result_Info;
    Keyboard_Info keyboard_info;

    HandSeg handseg;
    handseg.load_hand_model();

    int count_frame=0;
    // int_
    Mat base_img,image;

    //这个值可以调整，认为当前图像的亮度与整个视频的平均亮度相差小于20时此次为键盘亮度较好的图像
    //有些数据集前面是黑的或者光线不好，有些前面是一些其他东西，很亮
    int light_thresh=20;   //--可以适当调大，video中钢琴图像是最多的，因此第一张钢琴图像亮度接近平均亮度
    float ave_light=get_light(fn);
    Rect rect;
    for(std::size_t i=0;i<fn.size();i++){
        count_frame++;
        std::string path=fn[i];
	    string::size_type iPos = path.find_last_of("/") + 1;
	    string filename = path.substr(iPos, path.length() - iPos);
        // if (filename!="0156.jpg"){
        //     continue;
        // }

        Mat gray_img=imread(path,0);  //--for gray
        Scalar scalar=mean(gray_img);     

        //---还需要解决一个问题，有的视频最后才出现没有手的图像，如果从头开始遍历的话需要时间较长
        //--如果整个图像中钢琴上面都有手的话可以通过插值之类的将手那个区域轮廓检测出来然后补上黑键和白键的像素0.0

        //---对于网上下载的视频数据不合适，因为那种一开头反而更亮，与论文的数据集不一样，钢琴图片还没这么亮
        //----可以采用策略统计出所有图片的亮度，得到一个出现最多次数的亮度，如果当前图片亮度与该亮度相差<20则开始坚持

        //---排除掉路径不包含.jpg和图像亮度不好的图片,要选择较好的背景图，这样有利于检测黑/白键
        if((filename.find(".jpg")>1000)||(abs(scalar.val[0]-ave_light)>light_thresh)){
            continue;
        }


        image=imread(fn[i]);
        keyboard_info=keyboard.detect_keyboard(image);
        if (keyboard_info.flag==false){
            continue;
        }
        cout<<"the first detect keyboard img is "<<fn[i]<<endl;        
        rect=keyboard_info.keyboard_rect;
        Mat image_copy=image.clone();

        // int num_pixels=0;
        if (keyboard_info.rote_M.cols==0){
            base_img=image_copy(rect);
            result_Info.base_img=base_img.clone();
            result_Info.img=image_copy.clone();
            result_Info.count_frame=count_frame-1;
            result_Info.rect=rect;
            // num_pixels =handseg.detect_hand(image_copy,rect);
        }
        else{
            Mat rote_M=keyboard_info.rote_M.clone();
            if (keyboard_info.warp_M.cols==0){
                Mat rotated_img=keyboard_info.rotated_img.clone();
                base_img=rotated_img(rect);
                result_Info.base_img=base_img.clone();
                result_Info.img=rotated_img.clone();
                result_Info.count_frame=count_frame-1;
                result_Info.rect=rect;
                result_Info.rote_M=rote_M.clone();
                // num_pixels =handseg.detect_hand(rotated_img,rect);
            }
            else{
                Mat warp_M=keyboard_info.warp_M.clone();
                Mat warp_img=keyboard_info.warp_img.clone();
                base_img=warp_img(rect);
                result_Info.base_img=base_img.clone();
                result_Info.img=warp_img.clone();
                result_Info.count_frame=count_frame-1;
                result_Info.rect=rect;
                result_Info.rote_M=rote_M.clone();
                result_Info.warp_M=warp_M.clone();     
                // num_pixels =handseg.detect_hand(warp_img,rect); 
                //----for test----
            }
        }
        break;
    }


//--------------------------------------------------------------------------------------
    //---得到了rect,判断当前图像是不是背景，有时候背景在最后面，如果从头开始遍历耗时
    //---找到第一张能检测到36个黑键的图像，也可以设定开始遍历，如果从头开始几十张图像都没找到则从后面开始找
    int nums_thresh=0,nums=0;
    int Tostop=20;
    for(std::size_t i=0;i<fn.size();i++){    
        if(i<result_Info.count_frame){  //#--从第一个检测到rect的图片开始算
            continue;
        }
        nums_thresh++;

        std::string path=fn[i];
        //---顺着遍历30次还没找到则反向遍历背景图
        // cout<<"the nums_thresh is "<<nums_thresh<<endl;
        if(nums_thresh>Tostop){
            cout<<"the index is "<<int(fn.size()-nums-1)<<endl;
            path=fn[int(fn.size()-nums-1)];
            nums++;
        }        

        Mat image=imread(path);
        Mat image_copy=image.clone();
        if (keyboard_info.rote_M.cols==0){
            base_img=image_copy(rect);
        }
        else{
            Mat rote_M=keyboard_info.rote_M.clone();
            Mat rotated_img;
            warpAffine(image, rotated_img, rote_M, image.size());            
            if (keyboard_info.warp_M.cols==0){
                base_img=rotated_img(rect);
            }
            else{
                Mat warp_M=keyboard_info.warp_M.clone();
                Mat warp_img;      
                warpPerspective(rotated_img, warp_img, warp_M, image.size());
                base_img=warp_img(rect);
            }
        }

        //----直到第一个能检测到键盘的图像(可能包含手)
        Mat ori_img=base_img.clone();
        int height=ori_img.rows,width=ori_img.cols;
        vector<int> black_loc;
        
        //--新的find_black_boxes1没有对阈值进行多次设定,速度会快点
        // vector<Rect>black_boxes=find_black_boxe1s(ori_img,black_loc);
        vector<Rect>black_boxes=find_black_boxes(ori_img,black_loc);

        if (black_boxes.size()!=36){
            Mat blank=Mat::zeros(height,width,ori_img.type());
            double c=1.3,b=3;  //要看清楚函数传入的参数类型,double/int/float各不相同，不然会出错
            Mat dst_img;
            addWeighted(ori_img,c,blank,1-c,b,dst_img);
            black_boxes=find_black_boxes(dst_img,black_loc);
        }

        if(black_boxes.size()==37){
            int area1=black_boxes[0].width*black_boxes[0].height;
            int area2=black_boxes.back().width*black_boxes.back().height;
            //----有可能裁剪的钢琴区域左/右边会出现一点黑色区域，检测出来了
            if (area1>area2){
                black_boxes.erase(black_boxes.end());
            }
            else{
                black_boxes.erase(black_boxes.begin());
            }
        }
        // cout<<"cur img is "<<path<<endl;
        // cout<<"the black boxes is "<<black_boxes.size()<<endl;
        if (black_boxes.size()!=36){
            continue;
        }
        else{
            // for(size_t j=0;j<black_boxes.size();j++){
            //     rectangle(ori_img,black_boxes[j],Scalar(0,0,255),1);

            // }
            // imwrite("./black.jpg",ori_img);
            result_Info.base_img=base_img.clone();
            cout<<"the base frame is "<<path<<endl;
            break;
        }
    }

    return result_Info;
}

Mat remove_region(Mat &img){
    if (img.channels()==3){
        cout<<"please input a gray image"<<endl;
    }
    int h=img.rows,w=img.cols;
    for (int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            if((i<0.08*h  || i>(2.0/3)*h)||(j<0.005*w || j>0.994*w)){
                img.at<uchar>(i,j)=255;
            }
        }
    }
    // imwrite("./remove.jpg",img);
    return img;
}

vector<Rect> find_black_keys(Mat &base_img){
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
    base_img.convertTo(base_img,CV_8UC1);
	findContours(base_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());    
    int height=base_img.rows,width=base_img.cols;

    vector<cv::Rect> black_boxes;
    for(std::size_t k=0;k<contours.size();k++){
        Rect boundRect = boundingRect(Mat(contours[k]));
        int x=boundRect.x,y=boundRect.y;
        int w=boundRect.width,h=boundRect.height;
        if(h>height*0.3 &&w>4){
            int x1=x,y1=y,x2=x+w,y2=y+h;
            for(int i=y2;i>y1;i--){
                int count=0;
                for (int j=x1;j<x2;j++){
                    if (int(base_img.at<uchar>(i,j))!=0){
                        count++;
                    }
                }
                if(count>(x2-x1)*0.5){
                    black_boxes.push_back(Rect(x1,y1,w,i-y1));
                    break;
                }
            }
        }
    }
    
    if (black_boxes.size()!=36){
        vector<int>loc_width;
        for(std::size_t i=0;i<black_boxes.size();i++){
            loc_width.push_back(black_boxes[i].width);
        }
        vector<int>loc_width1(loc_width);//loc_width排序后顺序乱了
        std::sort(loc_width.begin(),loc_width.end());
        std::size_t idx=loc_width.size()/2;
        float middle_w;
        if (loc_width.size()%2==0){
            middle_w=(loc_width[idx]+loc_width[idx-1])/2.0;
        }
        else{
            middle_w=loc_width[idx];
        }
        vector<int> del_idx;
        for(std::size_t i=0;i<loc_width1.size();i++){
            if(loc_width1[i]<middle_w*0.5){
                del_idx.push_back(i);  //需要删除的元素
            }
        }
        vector<Rect> new_black_boxes;
        for(std::size_t i=0;i<loc_width1.size();i++){
            vector<int>::iterator it=find(del_idx.begin(),del_idx.end(),i);
            if(it==del_idx.end()){
                new_black_boxes.push_back(black_boxes[i]);
            }
        }
        // cout<<"the middle_w is "<<middle_w<<endl;
        return new_black_boxes;
    }
    return black_boxes;
}


bool rect_compare(Rect i, Rect j) {
	return (i.x < j.x);
}


vector<Rect> find_black_boxes1(Mat &ori_img,vector<int> &black_loc){
    int thresh=125;
    vector<Rect> black_boxes;
   
    Mat base_img=ori_img.clone();
    // int height=ori_img.rows,width=ori_img.cols;
    cvtColor(base_img,base_img,COLOR_BGR2GRAY);
    base_img=remove_region(base_img);
    threshold(base_img, base_img, thresh, 255, THRESH_BINARY_INV);
    black_boxes=find_black_keys(base_img);
    //--按照横坐标进行排序
    sort(black_boxes.begin(), black_boxes.end(), rect_compare);
    for(std::size_t i=0;i<black_boxes.size();i++){
        black_loc.push_back(black_boxes[i].x);
    }
    return black_boxes;
}


vector<Rect> find_black_boxes(Mat &ori_img,vector<int> &black_loc){
    int thresh=125;
    vector<Rect> black_boxes;
   
    while(true){
        
        black_loc.clear();  //每次循环前需要清0
        Mat base_img=ori_img.clone();
        // int height=ori_img.rows,width=ori_img.cols;
        cvtColor(base_img,base_img,COLOR_BGR2GRAY);
        base_img=remove_region(base_img);
        threshold(base_img, base_img, thresh, 255, THRESH_BINARY_INV);
        black_boxes=find_black_keys(base_img);
        //--按照横坐标进行排序
        sort(black_boxes.begin(), black_boxes.end(), rect_compare);
        for(std::size_t i=0;i<black_boxes.size();i++){
            black_loc.push_back(black_boxes[i].x);
        }
        if (black_loc.size()>36){
            thresh--;
        }
        else if(black_loc.size()<36){
            thresh++;
        }
        else{
            break;
        }
        if(thresh<90 || thresh>150){
            break;
        }
    }
    return black_boxes;
}

vector<float> find_white_loc(vector<int> &black_loc,vector<Rect> &black_boxes,const int &width){
    vector<float>white_loc;
    int black_gap1=black_loc[3]-black_loc[2];
    double ratio=23.0/41;
    double whitekey_width1 = ratio * black_gap1;
    int half_width1 = black_boxes[4].width;    //T1中第四个黑键被均分,从该位置开始算区域起始位置
    double keybegin = black_loc[4] + double(half_width1) / 2.0-7.0 * whitekey_width1;
    for(std::size_t i=0;i<10;i++){
        if(keybegin+i*whitekey_width1<0){
            white_loc.push_back(1);
        }
        else{
            white_loc.push_back(keybegin+i*whitekey_width1);
        }
    }
    for(std::size_t i=0;i<6;i++){
        int axis=8+i*5;
        int black_gap2 = black_loc[axis] - black_loc[axis - 1];
        double whitekey_width2 = ratio * black_gap2;
        int half_width2 = black_boxes[axis+1].width;
        double keybegin1 = black_loc[axis+1] + double(half_width2) / 2.0-5.0 * whitekey_width2;
        for(int j=1;j<8;j++){
            white_loc.push_back(keybegin1+j*whitekey_width2);
        }
        if(i==5){
            white_loc.push_back(min(width-1,int(keybegin1 + 8 * whitekey_width2)));
        }
    }
    return white_loc;
}

int near_white(const float &white_loc,const vector<Rect> &black_boxes){
    vector<float>diffs;
    for(std::size_t i=0;i<black_boxes.size();i++){
        float diff=abs(black_boxes[i].x-white_loc);
        diffs.push_back(diff);
    }
    auto min_diff=min_element(diffs.begin(),diffs.end());
    std::size_t index=distance(begin(diffs),min_diff);
    return index;
}

vector<int> white_black_dict(){
    vector<int>wh_dict;
    wh_dict.push_back(0); //--这个没用的
    wh_dict.push_back(0);
    wh_dict.push_back(0);
    for(std::size_t i=3;i<53;i++){
        int div=i/7;
        if (i%7==3 || i%7==4){
            wh_dict.push_back(div*5+1);
        }
        else if(i%7==5){
            wh_dict.push_back(div*5+2);
        }
        else if(i%7==6){
            wh_dict.push_back(div*5+3);
        }
        else if(i%7==0){
            wh_dict.push_back((div-1)*5+3);
        }
        else if(i%7==1){
            wh_dict.push_back((div-1)*5+4);
        }        
        else{
            wh_dict.push_back((div-1)*5+5);
        }
    }
    return wh_dict;
}

void find_key_loc(Mat &base_img,vector<Rect> &total_top,vector<Rect> &total_bottom, \
                  vector<Rect> &black_boxes,vector<float> &white_loc){
    Mat ori_img=base_img.clone();
    int height=ori_img.rows,width=ori_img.cols;
    vector<int> black_loc;
    
    black_boxes=find_black_boxes(ori_img,black_loc);
    if (black_boxes.size()!=36){
        Mat blank=Mat::zeros(height,width,ori_img.type());
        double c=1.3,b=3;  //要看清楚函数传入的参数类型,double/int/float各不相同，不然会出错
        Mat dst_img;
        addWeighted(ori_img,c,blank,1-c,b,dst_img);
        black_boxes=find_black_boxes(dst_img,black_loc);
    }

    if(black_boxes.size()==37){
        int area1=black_boxes[0].width*black_boxes[0].height;
        int area2=black_boxes.back().width*black_boxes.back().height;
        //----有可能裁剪的钢琴区域左/右边会出现一点黑色区域，检测出来了
        if (area1>area2){
            black_boxes.erase(black_boxes.end());
        }
        else{
            black_boxes.erase(black_boxes.begin());
        }
    }
    assert (black_boxes.size()==36);

    white_loc=find_white_loc(black_loc,black_boxes,width);

    Rect top_box,bottom_box;

    vector<int> wh_dict=white_black_dict();
    for(std::size_t i=1;i<white_loc.size();i++){
        float white_x = white_loc[i - 1];
        float white_width = white_loc[i] - white_x;
        std::size_t index=wh_dict[i];
        if((((i%7== 3) || (i%7==6)) && i < 52) || i==1){
            top_box=Rect(white_x, 0, max(int(black_boxes[index].x - white_x),int(1)), 1.1 * black_boxes[index].height); //---(x,y,w,h);
            bottom_box=Rect(white_x,1.1*black_boxes[index].height,white_width,height-1.1*black_boxes[index].height);
        }
        else if(i%7==4 || i%7==0 || i%7==1){
            top_box=Rect(black_boxes[index].x+black_boxes[index].width, 0, max(int(black_boxes[index+1].x - (black_boxes[index].x+black_boxes[index].width)),1), 1.1 * black_boxes[index].height);
            bottom_box=Rect(white_x,1.1*black_boxes[index].height,white_width+2,height-1.1*black_boxes[index].height);
        }
        else if(i%7==5 || i%7==2 || i==2){
            top_box=Rect(black_boxes[index].x+black_boxes[index].width, 0, white_loc[i] - max(int(black_boxes[index].x+black_boxes[index].width),1), 1.1 * black_boxes[index].height);
            bottom_box=Rect(white_x,1.1*black_boxes[index].height,white_width+2,height-1.1*black_boxes[index].height);
        }
        else {
            top_box=Rect(white_x + 1, 0, max(int(white_loc[i] - white_x - 1),1), 1.1 * black_boxes[35].height);
            bottom_box=Rect(white_x + 1, 1.1 * black_boxes[35].height, white_loc[i] - white_x - 1, height - 1.1 * black_boxes[35].height);
        }
       total_top.push_back(top_box);
       total_bottom.push_back(bottom_box);
    }
    for (std::size_t i=0;i<total_top.size();i++){
        total_top[i].width+=2;  //for draw
        rectangle(ori_img,total_top[i],Scalar(0,0,255),1);
        rectangle(ori_img,total_bottom[i],Scalar(0,0,255),1);
        // cout<<i<<endl;
        // cout<<total_top[i]<<endl;
    }
    imwrite("./res_0.jpg",ori_img);
}