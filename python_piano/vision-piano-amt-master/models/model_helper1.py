#-*- coding:utf-8 -*-
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
from torch.autograd import Variable
from PIL import Image
import cv2
import sys 
import os 
import numpy as np 
import time 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,PROJECT_ROOT)
from config import cfg 
from .hand_model import build_s3fd 
from .simple import SimpleNet
from .resnet_112_32 import ResNet18 as ResNet18_112 
from .conv3net import Conv3Net
from IPython import embed 

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


class ModelProduct(object):
    def __init__(self):
        self.load_det_hand_model() 
        
    def load_det_hand_model(self):
        self.hand_net = build_s3fd('test',5)
        self.hand_net.load_state_dict(torch.load('./seg_hand_resnet18.pth'))
        self.hand_net.eval()
        embed()
        if torch.cuda.is_available():
            self.hand_net.cuda()
        cudnn.benchmark = True 


    def detect_hand(self,img,Rect):
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img)
        height,width,_ = img.shape 
        img = img[Rect[1]:height,:]
        ori_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        max_im_shrink = np.sqrt(720 * 640 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink,fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
        x = to_chw_bgr(image).astype('float32')
        img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
        x -= img_mean
        x = x[[2, 1, 0], :, :]
        
        x = Variable(torch.from_numpy(x).unsqueeze(0))
        x = x.cuda()
        y = self.hand_net(x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        hand_box = []
        for i in range(detections.size(1)):
            j = 0
            #----j=0表示的是取出当前类预测boxes分数最高的那个boxes0，按照nms排序的
            #----最后一个维度的0表示的是取出scors
            while detections[0, i, j, 0] >= cfg.VIS_THRESH:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()
                lr_mark = detections[0,i,j,5:].cpu().numpy()
                left_up, right_bottom = (int(pt[0]), int(pt[1]+Rect[1])), (int(pt[2]), int(pt[3]+Rect[1]))
                hand_box.append((left_up,right_bottom))
                j += 1
        return hand_box
        
