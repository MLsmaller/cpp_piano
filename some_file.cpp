

Keyboard_Info KeyBoard::test_post_process2(Mat &image,Mat save_mask){
    Keyboard_Info keyboard_info;
    ori_size=cv::Size(image.cols,image.rows);
    
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST);

    // int w=save_mask.cols,h=save_mask.rows;

	cvtColor(save_mask, save_mask, COLOR_BGR2GRAY);
	threshold(save_mask, save_mask, 150, 255, THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(save_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());

	int max_index = 0;
	vector<Point>board_contours;
	//--由于contours.size()返回的是无符号数，这里定义为size_t而不是Int，不然有警告
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

Keyboard_Info KeyBoard::test_post_process(Mat &image,Mat save_mask){
    Keyboard_Info keyboard_info;
    ori_size=cv::Size(image.cols,image.rows);
    
    resize(save_mask,save_mask,ori_size,0,0,INTER_NEAREST);

    int w=save_mask.cols,h=save_mask.rows;

	cvtColor(save_mask, save_mask, COLOR_BGR2GRAY);
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
		rt = keyboard_rect[2]; rb = keyboard_rect[3];
        if (abs(lt.y - rt.y) > 5 || abs(rb.y - lb.y) > 5) {
			int xb1 = lb.x, yb1 = lb.y, xb2 = rb.x, yb2 = rb.y;
			int xt1 = lt.x, yt1 = lt.y, xt2 = rt.x, yt2 = rt.y;
			Point2f center = Point2f(w / 2, h / 2);
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
				imwrite("./c_rota.jpg",rotated_img);
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