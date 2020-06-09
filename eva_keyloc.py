import cv2
import numpy as np
import os
import sys
sys.path.append('./python_piano/vision-piano-amt-master/piano_utils')

from util import calAngle,order_points 
from IPython import embed

def remove_region(img):
    if len(img.shape) == 3:
        print("please input a gray image")
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if (i < 0.08 * h or i > (2.0/3) * h):
                img[i, j] = 255
    for i in range(h):
        for j in range(w):
            if (j < 0.005 * w or j > 0.994 * w):
                img[i, j] = 255
    return img


def find_black_keys(base_img):
    _,contours,_ = cv2.findContours(base_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    black_boxes = []
    height,width = base_img.shape[:2]
    for idx,cnt in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(cnt)
        if h>height*0.3 and w>4:
            x1,y1,x2,y2 = x,y,x+w,y+h 
            for i in range(y2,y1,-1):
                count = 0 
                for j in range(x1,x2):
                    if base_img[i,j]!= 0:
                        count+=1 
                if count > (x2-x1)*0.5:
                    black_boxes.append((x1,y1,w,i-y1))
                    break 
    # for i in range(len(black_boxes)):
    #     print(black_boxes[i])
    #     print('\n')

    if len(black_boxes)!=0:
        ws = [box[2] for box in black_boxes]
        ws = np.array(ws)
        me = np.median(ws)
        print('the me is {}'.format(me))
        for i,wd in enumerate(ws):
            if wd<me*0.5:del black_boxes[i]
    return black_boxes


def find_black_boxes(ori_img):
    thresh = 125
    while True:
        base_img = ori_img.copy()
        height,width,_ = base_img.shape 
        base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
        base_img = remove_region(base_img)
        _,base_img = cv2.threshold(base_img,thresh,255,cv2.THRESH_BINARY_INV)
    
        black_boxes = find_black_keys(base_img)
        black_boxes = sorted(black_boxes,key = lambda x:x[0])
        black_loc = [box[0] for box in black_boxes]
        if len(black_loc)>36:
            thresh-=1
        elif len(black_loc)<36:
            thresh+=1
        else:
            break
        if thresh<90 or thresh>150:
            break
    return black_boxes,black_loc 

def contrast_img(img, c, b):
    rows, cols, channels = img.shape 
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return  dst 

def find_white_loc_old(black_loc,black_boxes,width):
    white_loc = []
    black_gap1 = black_loc[3] - black_loc[2]  #--第一个周期区域内的黑键间隔
    ratio = 23.0 / 41
    # ratio = 23.0 / 40
    whitekey_width1 = ratio * black_gap1  
    half_width1 = black_boxes[4][2]    #T1中第四个黑键被均分,从该位置开始算区域起始位置
    keybegin = black_loc[4] + half_width1 / 2.0-7.0 * whitekey_width1
    for i in range(10):
        # print(keybegin + i * whitekey_width1)

        if keybegin + i * whitekey_width1 < 0:
            white_loc.append(1)
        else:
            white_loc.append(keybegin + i * whitekey_width1)

    for i in range(6):  #----剩下的6个循环区域
        axis = 8 + i * 5
        black_gap2 = black_loc[axis] - black_loc[axis - 1]
        whitekey_width2 = ratio * black_gap2 
        half_width2 = black_boxes[axis + 1][2] 
        keybegin1 = black_loc[axis + 1] + float(half_width2 / 2.0) - 5.0 * whitekey_width2
        for j in range(1,8):
            white_loc.append(keybegin1 + j * whitekey_width2)
        if i == 5:  #----最后一次循环将钢琴最后一个白键加上
            if width < int(keybegin1 + 8 * whitekey_width2):
                white_loc.append(width - 1)
            else:
                white_loc.append(keybegin1 + 8 * whitekey_width2)
      
    return white_loc 

def near_white(white_loc,black_boxes):
    diffs = []
    for i in range(len(black_boxes)):
        diff = abs(black_boxes[i][0] - white_loc)
        diffs.append(diff)
    index = diffs.index(min(diffs))
    return index


def white_black_dict():
    wh_dict={}
    wh_dict[1]=0
    wh_dict[2]=0

    for i in range(3,53):
        div=int(i/7)
        if i%7==3 or i%7==4:
            wh_dict[i]=div*5+1
        elif i%7==5:
            wh_dict[i]=div*5+2
        elif i%7==6:
            wh_dict[i]=div*5+3
        elif i%7==0:
            wh_dict[i]=(div-1)*5+3
        elif i%7==1:
            wh_dict[i]=(div-1)*5+4
        else :
            wh_dict[i]=(div-1)*5+5
    return wh_dict


def find_key_loc(base_img):
    white_loc = []
    black_boxes = []
    total_top = []
    total_bottom = [] 
    black_loc = []    
    ori_img = base_img.copy()
    draw_img = base_img.copy()
    height,width,_ = ori_img.shape 
    black_boxes,black_loc = find_black_boxes(ori_img)
    if len(black_boxes)!=36:
        ori_img = contrast_img(ori_img,1.3,3)
        cv2.imwrite('./py_img.jpg',ori_img)
        black_boxes,black_loc = find_black_boxes(ori_img)
        
    if len(black_boxes)==37:
        area1 = black_boxes[0][2]*black_boxes[0][3]
        area2 = black_boxes[-1][2]*black_boxes[-1][3]
        if area1>area2:
            del black_boxes[-1]
        else:
            del black_boxes[0]

    for i in range(len(black_boxes)):
        x1,y1=black_boxes[i][0],black_boxes[i][1]
        x2,y2=x1+black_boxes[i][2],y1+black_boxes[i][3]
        cv2.rectangle(ori_img,(x1,y1),(x2,y2),(0,0,255),1)
        # print(i,'\n')
        # print(total_top[i])
    cv2.imwrite("./py_black.jpg",ori_img)
    assert len(black_boxes)==36,'black number is wrong'

    white_loc=find_white_loc_old(black_loc,black_boxes,width)

    # for i in range(len(white_loc)):
    #     x=int(white_loc[i])
    #     # cv2.line(ori_img,(x,0),(x,height),(0,0,255),2)
    #     cv2.putText(ori_img,str(i+1),(int(x+1),height-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1,cv2.LINE_AA)

    # cv2.imwrite('./py_line.jpg',ori_img)

    wh_dict=white_black_dict()

    for i in range(1, len(white_loc)):
        white_x = white_loc[i - 1]
        white_width = white_loc[i] - white_x
        index=wh_dict[i]
        if (((i%7== 3) or (i%7==6)) and i < 52) or i==1:
            top_box = (white_x, 0, black_boxes[index][0] - white_x, 1.1 * black_boxes[index][3]) #---(x,y,w,h)
            bottom_box=(white_x,1.1*black_boxes[index][3],white_width,height-1.1*black_boxes[index][3])
        elif i%7==4 or i%7==0 or i%7==1:
            top_box = (black_boxes[index][0]+black_boxes[index][2], 0, black_boxes[index+1][0] - (black_boxes[index][0]+black_boxes[index][2]) - 1, 1.1 * black_boxes[index][3])
            bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
        elif i%7==5 or i%7==2 or i==2:
            top_box = (black_boxes[index][0]+black_boxes[index][2], 0, white_loc[i] - (black_boxes[index][0]+black_boxes[index][2]) - 1, 1.1 * black_boxes[index][3])
            bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
            
     #----最后一个框
        else:
            top_box = (white_x + 1, 0, white_loc[i] - white_x - 1, 1.1 * black_boxes[35][3])
            bottom_box = (white_x + 1, 1.1 * black_boxes[35][3], white_loc[i] - white_x - 1, height - 1.1 * black_boxes[35][3])
            
        total_top.append(top_box)
        total_bottom.append(bottom_box)


    white_loc = np.array(white_loc,dtype=np.int32)
    black_boxes = np.array(black_boxes,dtype=np.int32)
    total_top = np.array(total_top,dtype=np.int32)
    total_bottom = np.array(total_bottom,dtype=np.int32)    
    for i in range(len(total_top)):
        x1,y1=total_top[i][0],total_top[i][1]
        x2,y2=x1+total_top[i][2],y1+total_top[i][3]
        p1,k1=total_bottom[i][0],total_bottom[i][1]
        p2,k2=p1+total_bottom[i][2],k1+total_bottom[i][3]

        cv2.rectangle(ori_img,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.rectangle(ori_img,(p1,k1),(p2,k2),(0,0,255),1)
        # print(i,'\n')
        # print(total_top[i])
    cv2.imwrite('./py_res.jpg',ori_img)


if __name__=='__main__':

    img_paths='./keyboard_images/rect'
    img_paths=[os.path.join(img_paths,x) for x in os.listdir(img_paths)
            if x.endswith(('.png','.jpg'))]

    for img_path in img_paths:
        if not os.path.basename(img_path)=='rect_mask0004.png': continue
        print(img_path)
        img=cv2.imread(img_path)
        find_key_loc(img)

        break
        

