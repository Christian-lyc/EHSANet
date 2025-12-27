import pydicom
import argparse
import os
import torch
from dataset import DenoiseDataset
import torchvision.transforms as transforms
#from model import SwinUNet
#import horovod.torch as hvd
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.functional import peak_signal_noise_ratio
import numpy as np
#from unet import UNet
from model import SwinUNet
from measure import compute_measure
from loader import get_loader

parser = argparse.ArgumentParser(description='PyTorch LDCT denoising')
parser.add_argument('--epochs',default=1000,type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--weight_decay',default=1e-4,type=float)
parser.add_argument('--data',default='/export/data/.../PET',type=str)
parser.add_argument('--patch_n', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160.0)
parser.add_argument('--trunc_max', type=float, default=240.0)
args = parser.parse_args()

trunc_min=args.trunc_min
trunc_max=args.trunc_max
norm_range_max=args.norm_range_max
norm_range_min=args.norm_range_min

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def denormalize_( image):
        image = image * (norm_range_max - norm_range_min) + norm_range_min
        return image

def trunc( mat):
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat

def sobel_filter(image_tensor):  # shape: (1, 1, H, W)
    sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)
    sobel_kernel_xy= torch.tensor([[[0, 1, 2], [-1, 0, 1], [-2,-1, 0]]], dtype=torch.float32).unsqueeze(0)
    sobel_kernel_yx= torch.tensor([[[-2, -1, 0], [-1, 0, 1], [0,1, 2]]], dtype=torch.float32).unsqueeze(0)
    sobel_kernel_x = sobel_kernel_x.to(image_tensor.device)
    sobel_kernel_y = sobel_kernel_y.to(image_tensor.device)
    sobel_kernel_xy= sobel_kernel_xy.to(image_tensor.device)
    sobel_kernel_yx= sobel_kernel_yx.to(image_tensor.device)

    Gx = F.conv2d(image_tensor, sobel_kernel_x, padding=1)
    Gy = F.conv2d(image_tensor, sobel_kernel_y, padding=1)
    Gxy = F.conv2d(image_tensor, sobel_kernel_xy, padding=1)
    Gyx = F.conv2d(image_tensor, sobel_kernel_yx, padding=1)

    G = torch.sqrt(Gxy**2+Gyx**2+Gx ** 2 + Gy ** 2)
    return G



def main():

    f='/export/home/.../LDCT/swin/save_PET/swin_unet.ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traindir = os.path.join(args.data, 'pet_train')
    valdir = os.path.join(args.data, 'test9')
    padding = (0, 0, 72, 72)
    transform = transforms.Compose([
        transforms.Pad(padding, fill=0),
        transforms.ToTensor(),
    ])


    test_loader = get_loader(mode='test',
                             load_mode=0,
                             saved_path=valdir,
                             patch_n= None,
                             patch_size=None,
                             transform=transform,
                             batch_size=1,
                             num_workers=args.num_workers)

    model = SwinUNet(64,64,1,32,num_blocks=2).cuda()
    model.load_state_dict(torch.load(f,map_location=torch.device('cuda:0')))
    NumOfParam = count_parameters(model)
    print('trainable parameter:', NumOfParam)

    criterion = nn.L1Loss().cuda()  #nn.MSELoss().cuda()

    iter_num=0
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    for low,full in test_loader:
        shape_ = low.shape[-1]
        model.eval()

        if len(low.shape)==3:
            full=full.unsqueeze(0).to(torch.float32).to(device)
            y=model(low.unsqueeze(0).to(torch.float32).to(device)).detach()
        else:
            low = low.to(torch.float32).to(device)
            full= full.to(torch.float32)
            y=model(low).detach()
      
        low =low.view(shape_, shape_)[:440,:440].cpu().detach()*255
        full=full.view(shape_, shape_)[:440,:440].cpu().detach()*255
        y=y.to(torch.float32).view(shape_, shape_)[:440,:440].cpu().detach()*255
        data_range =255

        original_result, pred_result = compute_measure(low, full, y, data_range)
        ori_psnr_avg += original_result[0]
        ori_ssim_avg += original_result[1]
        ori_rmse_avg += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]

    print('\n')

    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(test_loader),
                                                                                            ori_ssim_avg/len(test_loader),
                                                                                            ori_rmse_avg/len(test_loader)))
    print('\n')
    print('epoch: \n Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(test_loader),
                                                                                                  pred_ssim_avg/len(test_loader),
                                                                                                  pred_rmse_avg/len(test_loader)))





if __name__ == '__main__':
    main()


