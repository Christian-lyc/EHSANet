import matplotlib.pyplot as plt
import numpy as np
import pydicom
#from CTformer import CTformer
#from unet import UNet
#from networks import RED_CNN
from model import SwinUNet
#from network_swinir import SwinIR as net
import torch
import torch.nn as nn
#import cv2
LDPET=np.load('/export/home/.../LDCT/low_dose_181022_6_2956.npy')
FDPET=np.load('/export/home/.../LDCT/full_dose_181022_6_2956.npy')

trunc_min=-160.0
trunc_max=240.0
norm_range_max=3072.0
norm_range_min=-1024.0
def trunc( mat):
    mat[mat <= trunc_min] = trunc_min
    mat[mat >= trunc_max] = trunc_max
    return mat

def denormalize_( image):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image

def window_image(img, window_min=-160, window_max=240):
    img = np.clip(img, window_min, window_max)
    img = ((img - window_min) / (window_max - window_min)) * 255
    return img.astype(np.uint8)

def split_arr(arr,patch_size,stride=32):    ## 512*512 to 32*32
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
  #return arr
    return arr.unsqueeze(0).unsqueeze(1)

"""
def draw_arrows(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Draw arrows manually at two positions (x1, y1), (x2, y2)
    img_rgb = cv2.arrowedLine(img_rgb, (150, 350), (200, 350), (255, 0, 0), 4, tipLength=0.3)
    img_rgb = cv2.arrowedLine(img_rgb, (300, 100), (350, 100), (255, 0, 0), 4, tipLength=0.3)
    return img_rgb
"""

f='/export/home/.../LDCT/swin1/save_PET1/swin_unet_2999iter7.ckpt' #PET

model = SwinUNet(64,64,1,32,num_blocks=2)
model.load_state_dict(torch.load(f))
model.eval()

LDPET = np.pad(LDPET, ((0, 72), (0, 72)), mode='constant', constant_values=0)
print(LDPET.shape)
input=torch.from_numpy(LDPET).unsqueeze(0).unsqueeze(0).to(torch.float32)
shape_ = input.shape[-1]


y=model(input)

low =LDPET[:440,:440].reshape(440, 440)*255
full=FDPET.reshape(440, 440)*255
y=y.view(shape_, shape_).cpu().detach().numpy()*255
pred=y[:440,:440]
images=[low,pred,full]
windowed_images = [window_image(img) for img in images]

plt.imshow(windowed_images[1], cmap='gray')
plt.title('HSANet', fontsize=12)


plt.tight_layout()
plt.savefig('PET_HSANet.png', dpi=500)
#plt.show()
