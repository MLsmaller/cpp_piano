import cv2
import numpy as np
import os
import sys
sys.path.append('./python_piano/vision-piano-amt-master/piano_utils')

from util import calAngle,order_points 
from IPython import embed


def find_contours(pmask):
    h,w = pmask.shape 
    _,base_img = cv2.threshold(pmask,150,255,cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(base_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    board_contours = np.array([])
    for contour in contours:
        contour = np.array(contour,dtype=np.int32)
        if len(contour)>len(board_contours):
            board_contours = contour
    contours = np.squeeze(board_contours)
    return contours,pmask 


def find_rect(pmask,sx,sy,ex,ey):
    height,width = pmask.shape
    loc_x,loc_y = [],[]
    for i in range(sy,ey):
        for j in range(sx,ex):
            if pmask[i,j]!=0:
                loc_y.append(i)
    loc_y.sort()
    loc_y = np.unique(np.array(loc_y))
    locy_min,locy_max = 0,0
    for y in loc_y:
        #----这里相当于是找出纵轴y的最小值
        cmask = np.where(pmask[y]!=0)[0]
        if len(cmask)>0.3*width:
            locy_min = y 
            break 
    for y in loc_y[::-1]:
        cmask = np.where(pmask[y]!=0)[0]
        if len(cmask)>0.3*width:
            locy_max = y 
            break 
    piano_ylen = locy_max-locy_min 
    locx_min,locx_max = 0,0
    for x in range(sx,ex):
        cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
        if len(cmask)>0.3*(piano_ylen):
            locx_min = x 
            break 
    for x in range(sx,ex)[::-1]:
        cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
        if len(cmask)>0.3*piano_ylen:
            locx_max = x 
            break
    Rect = (locx_min,locy_min,locx_max,locy_max)
    if locy_max-locy_min<20:
        return False,Rect 
    return True,Rect 

def process2(mask):
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    contours,pmask =find_contours(mask)
    result = {}
    rect = order_points(contours)
    if len(contours)>500:
        lt,rt,rb,lb = rect
        sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
        sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
        flag,keyboard_rect = find_rect(pmask,sx,sy,ex,ey)
        result = {'flag':flag,'keyboard_rect':keyboard_rect}
    else:
        result = {'flag':0,'keyboard_rect':None}
    return result     


if __name__=='__main__':

    img_paths='./mask_imgs'
    img_paths=[os.path.join(img_paths,x) for x in os.listdir(img_paths)
            if x.endswith(('.png','.jpg'))]

    for img_path in img_paths:
        if not os.path.basename(img_path)=='post1_0000.png': continue
        print(img_path)
        mask=cv2.imread(img_path)
        result=process2(mask)
        rect=result['keyboard_rect']
        cv2.rectangle(mask,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),3) 
        cv2.imwrite("./py_rect.png",mask)
        break
        


# img=cv2.imread('./0146.jpg')
# mask=cv2.imread('./mask.png')
# mask=cv2.resize(mask,(img.shape[1],img.shape[0]),cv2.INTER_NEAREST)




# mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# contours,pmask =find_contours(mask)

# result = {}
# rect = order_points(contours)
# h,w=pmask.shape
# if len(contours)>500:
#     lt,rt,rb,lb = rect    
#     if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
#         xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
#         xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
#         center = (w//2,h//2)
#         if abs(yb1-yb2)>abs(yt1-yt2):
#             angle = calAngle(xb1,yb1,xb2,yb2)
#             M = cv2.getRotationMatrix2D(center,angle,1)
#             rotated_img = cv2.warpAffine(img,M,(w,h))
#         else:
#             angle = calAngle(xt1,yt1,xt2,yt2)
#             M = cv2.getRotationMatrix2D(center,angle,1)
#             rotated_img = cv2.warpAffine(img,M,(w,h))
#         result = {'flag':1,'rote_M':M,'warp_M':None,'keyboard_rect':None,
#                 'rotated_img':rotated_img 
#         }
#     else:
#         lr,rt,rb,lb = rect
#         sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
#         sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
#         flag,keyboard_rect = find_rect(pmask,sx,sy,ex,ey)
#         result = {
#                 'flag':flag,
#                 'rote_M':None,
#                 'warp_M':None,
#                 'keyboard_rect':keyboard_rect,
#                 'rotated_img':None 
#         }

# else:
#     result = {'flag':0,
#             'rote_M':None,
#             'warp_M':None,
#             'keyboard_rect':None,
#             'rotated_img':None}

# print(mask.shape)
