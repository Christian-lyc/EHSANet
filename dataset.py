import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torch
import pydicom
from glob import glob
import os


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = slices.pixel_array#np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    #for slice_number in range(len(slices)):
    intercept = slices.RescaleIntercept
    slope = slices.RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def normalize_(image, MIN_B=-1024.0, MAX_B=3072.0):
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image


class DenoiseDataset(Dataset):
    def __init__(self, traindir=None,valdir=None,transform=None):
        self.traindir = traindir
        self.valdir = valdir
        self.pairs=[]
        self.transform = transform
        if self.traindir:
            train_list = glob(os.path.join(self.traindir, '*','*','*'))
            full_dose_dirs = [d for d in train_list if os.path.isdir(d) and 'Full' in os.path.basename(d)]
            low_dose_dirs = [d for d in train_list if os.path.isdir(d) and 'Low' in os.path.basename(d)]
            full_dose_list=glob(os.path.join(full_dose_dirs[0],'*'))
            low_dose_list=glob(os.path.join(low_dose_dirs[0],'*'))
            if low_dose_list and full_dose_list:
                low_dose_list.sort()
                full_dose_list.sort()
                for l,f in zip(low_dose_list,full_dose_list):
                    self.pairs.append((l,f))

        if self.valdir:
            val_list = glob(os.path.join(self.valdir, '*','*','*'))
            full_dose_dirs = [d for d in val_list if os.path.isdir(d) and 'Full' in os.path.basename(d)]
            low_dose_dirs = [d for d in val_list if os.path.isdir(d) and 'Low' in os.path.basename(d)]
            full_dose_list=glob(os.path.join(full_dose_dirs[0],'*'))
            low_dose_list=glob(os.path.join(low_dose_dirs[0],'*'))
            if low_dose_list and full_dose_list:
                low_dose_list.sort()
                full_dose_list.sort()
                for l,f in zip(low_dose_list,full_dose_list):
                    self.pairs.append((l,f))



    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path,full_path = self.pairs[idx]
        low_dicom=pydicom.dcmread(low_path)
        full_dicom=pydicom.dcmread(full_path)
        low_dicom=normalize_(get_pixels_hu(low_dicom))
        full_dicom=normalize_(get_pixels_hu(full_dicom))
        low_tensor = torch.from_numpy(low_dicom).unsqueeze(0).float()
        full_tensor = torch.from_numpy(full_dicom).unsqueeze(0).float()
        return low_tensor,full_tensor
