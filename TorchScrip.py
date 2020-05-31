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
    image = image.resize((960, 600))
    input = normalize(to_tensor(image)).unsqueeze(0)
    #---读取进去的是rgb通道

    print(input.size())
    t1 = time.time()
    prediction = model(input.to(device))
    print('the prediction size is {}'.format(prediction.size()))


    prediction = prediction.squeeze(0).cpu().detach().numpy()
    print("img {} cost {} s".format(img_path,time.time()-t1))
    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()    
    

    test_mask=prediction.copy()
    test_mask[np.where(test_mask==1)]=255
    cv2.imwrite('./test_mask.png',test_mask)

    colorized_mask = colorize_mask(prediction, palette)
    pmask = np.array(colorized_mask)
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    _,contours, hier = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    for cidx,cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h>50:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    output_path='./save_path'
    image_file=os.path.basename(img_path).split('.')[0]
    cv2.imwrite(os.path.join(output_path,image_file+'.jpg'),image)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))    
    

path='./keyboard.pth'  #导入的模型文件必须与当前文件在同一目录下
checkpoint = torch.load(path)

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()  

img_path="./0146.jpg"
test_img(img_path,model)


# # An example input you would normally provide to your model's forward() method.
#--期待的输入数据--
example = torch.rand(1, 3, 600, 960).to(device)

# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("./keyboard1.pt")
