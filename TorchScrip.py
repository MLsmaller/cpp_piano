#-*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

from IPython import embed
from pspnet import PSPNet

from PIL import Image
import numpy as np
import cv2
import time
import os
import math

# An instance of your model.
model=PSPNet(num_classes=2)
x=torch.randn(1,3,224,224)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = [0.45734706, 0.43338275, 0.40058118]
STD = [0.23965294, 0.23532275, 0.2398498]

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(MEAN,STD)

num_classes = 2
palette = [0,0,0,128,0,128]

def calAngle( x1,  y1,  x2,  y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan(dy/dx)
    return (angle * 180 / math.pi)

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    #---lt,rt,rb,lb->rect
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)   
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def colorize_mask(mask, palette):
    #---相当于这里putpalette()函数是添加一个调色板，用于显示
    #---除了特定的palette中的值，其他的值都是0，因为这里只有两类(背景和手)
    #---对应的mask中的值分别是0/1，因此对应的值都为（0,0,0）(128,0,128)
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask



def test_img(img_path,model):
    
    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    #image = image.resize((960, 600))
    image = image.resize((960, 600))   #--for hand
    input = normalize(to_tensor(image)).unsqueeze(0)
    #---读取进去的是rgb通道

    print(input.size())
    t1 = time.time()
    prediction = model(input.to(device))
    print('the prediction size is {}'.format(prediction.size()))


    prediction = prediction.squeeze(0).cpu().detach().numpy()
    print("img {} cost {} s".format(img_path,time.time()-t1))
    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()    
    print(prediction)
    # print(prediction.shape)

    test_mask=prediction.copy()
    test_mask[np.where(test_mask==1)]=255
    cv2.imwrite('./test_mask.png',test_mask)

def test_mask(img,img_path):
    mask=cv2.imread(img_path)
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    h,w=mask.shape
    prediction=mask

    colorized_mask = colorize_mask(prediction, palette)
    pmask = np.array(colorized_mask)
    _,contours, hier = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    board_contours = np.array([])
    for contour in contours:
        contour = np.array(contour,dtype=np.int32)
        if len(contour)>len(board_contours):
            board_contours = contour
    contours = np.squeeze(board_contours)

    result = {}
    rect = order_points(contours)
    if len(contours)>500:
        lt,rt,rb,lb = rect            
        if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
            xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
            xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
            center = (w//2,h//2)
            if abs(yb1-yb2)>abs(yt1-yt2):
                angle = calAngle(xb1,yb1,xb2,yb2)
                M = cv2.getRotationMatrix2D(center,angle,1)
                rotated_img = cv2.warpAffine(img,M,(w,h))
            else:
                angle = calAngle(xt1,yt1,xt2,yt2)
                M = cv2.getRotationMatrix2D(center,angle,1)
                print(angle)
                print('axis is :')
                print((xt1,yt1,xt2,yt2))
                print(center)
                print(w,h)
                print(M)
                rotated_img = cv2.warpAffine(img,M,(w,h))
                cv2.imwrite('./rotated.jpg',rotated_img)
                embed()
            result = {'flag':1,'rote_M':M,'warp_M':None,'keyboard_rect':None,
                    'rotated_img':rotated_img 
            }
        else:
            lr,rt,rb,lb = rect
            sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
            sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
            flag,keyboard_rect = self.find_rect(pmask,sx,sy,ex,ey)
            result = {
                    'flag':flag,
                    'rote_M':None,
                    'warp_M':None,
                    'keyboard_rect':keyboard_rect,
                    'rotated_img':None 
            }

    else:
        result = {'flag':0,
                'rote_M':None,
                'warp_M':None,
                'keyboard_rect':None,
                'rotated_img':None}
    
    # for cidx,cnt in enumerate(contours):
    #     (x, y, w, h) = cv2.boundingRect(cnt)
    #     if h>50:
    #         cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # output_path='./save_path'
    # image_file=os.path.basename(img_path).split('.')[0]
    # cv2.imwrite(os.path.join(output_path,image_file+'.jpg'),image)
    # colorized_mask.save(os.path.join(output_path, image_file+'.png'))    
    

if __name__=='__main__':
    path='./keyboard.pth'  #导入的模型文件必须与当前文件在同一目录下
    # path='./seg_hand.pth'
    checkpoint = torch.load(path)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  
    paramerters=sum(x.numel() for x in model.parameters())
    #---model->51M
    print("models have {} M paramerters in total".format(paramerters/1e6))

    img_paths='./mask_imgs'
    img_paths=[os.path.join(img_paths,x) for x in os.listdir(img_paths)
            if x.endswith('.png')]

    # for img_path in img_paths:
    #     if not os.path.basename(img_path)=='0000.png':continue
    #     img=cv2.imread('./keyboard_images/0015.jpg')
    #     print(img_path)
    #     test_mask(img,img_path)


    # # An example input you would normally provide to your model's forward() method.
    #--期待的输入数据--
    #example = torch.rand(1, 3, 600, 960).to(device)  #for keyboard
    # example = torch.rand(1, 3, 600, 960).to(device)  #for hand

    # # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    # traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save("./hand_seg.pt")
