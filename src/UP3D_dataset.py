import torch
import torch.utils.data as data

import torchvision.transforms as transforms

import scipy.io
from PIL import Image
import os

from .utils import smart_padding, smart_padding_depth
import numpy as np
import cv2
import pickle as pkl


class UP3DDatasetGTSMRGB(data.Dataset):
    def __init__(self, dstype):
        super(UP3DDatasetGTSMRGB, self).__init__()

        self.root_dir = "/media/HDD_2TB/yudhik/up-3d/up-3d/"

        if dstype == "train":
            listfile = "/media/HDD_2TB/yudhik/up-3d/up-3d/train.txt"
        elif dstype == "val":
            listfile = "/media/HDD_2TB/yudhik/up-3d/up-3d/val.txt"

        with open(listfile, "r") as f:
            self.img_names = f.readlines()

        self.img_names = [nm.strip()[1:] for nm in self.img_names]
        self.length = len(self.img_names)

        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return self.length

    def get_image_name(self, idx):
        return self.img_names[idx]

    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        seg_name = img_name[:-9] + "render_light.png"

        crop_name = img_name[:5] + "_fit_crop_info.txt"
        pkl_name = img_name[:5] + "_body.pkl"

        with open(os.path.join(self.root_dir, crop_name) ) as f:
            crop_params = f.readlines()[0]
        crop_params = [int(ii) for ii in crop_params.strip().split()]

        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.crop((crop_params[4], crop_params[2], crop_params[5], crop_params[3]))
        img = self.tsfm(img).float()

        seg_path = os.path.join(self.root_dir, seg_name)
        seg = Image.open(seg_path).convert('RGB')
        seg = seg.crop((crop_params[4], crop_params[2], crop_params[5], crop_params[3]))
        seg = self.tsfm(seg).float()

        datapkl = pkl.load(open( os.path.join(self.root_dir, pkl_name), "rb" ), encoding='latin1')
        beta = torch.tensor(datapkl['betas'], dtype=torch.float)
        theta = torch.tensor(datapkl['pose'], dtype=torch.float)

        return img, seg, theta, beta



class UP3DDatasetDPSMRGB(data.Dataset):
    def __init__(self, dstype):
        super(UP3DDatasetDPSMRGB, self).__init__()

        self.root_dir = "/media/HDD_2TB/yudhik/up-3d/up-3d/"
        self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/up3d"

        if dstype == "train":
            listfile = "/media/HDD_2TB/yudhik/up-3d/up-3d/train.txt"
        elif dstype == "val":
            listfile = "/media/HDD_2TB/yudhik/up-3d/up-3d/val.txt"

        with open(listfile, "r") as f:
            self.img_names = f.readlines()

        self.img_names = [nm.strip()[1:] for nm in self.img_names]
        self.length = len(self.img_names)

        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        iuv_mat_name = img_name.split(".")[0] + "_IUV.mat"

        if not os.path.exists( os.path.join(self.dp_root_dir, iuv_mat_name) ):
            idx = 0
            img_name = self.img_names[idx]
            iuv_mat_name = img_name.split(".")[0] + "_IUV.mat"

        crop_name = img_name[:5] + "_fit_crop_info.txt"
        pkl_name = img_name[:5] + "_body.pkl"

        with open(os.path.join(self.root_dir, crop_name) ) as f:
            crop_params = f.readlines()[0]
        crop_params = [int(ii) for ii in crop_params.strip().split()]

        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.crop((crop_params[4], crop_params[2], crop_params[5], crop_params[3]))
        img = self.tsfm(img).float()

        segmask = scipy.io.loadmat(os.path.join(self.dp_root_dir, iuv_mat_name) )['segm']
        segmask = segmask[crop_params[2]:crop_params[3], crop_params[4]:crop_params[5] ]
        segmask = smart_padding_depth(segmask)
        segmask = cv2.resize(segmask, (224, 224))
        segmask = torch.tensor(segmask).unsqueeze(0).float()

        datapkl = pkl.load(open( os.path.join(self.root_dir, pkl_name), "rb" ), encoding='latin1')

        beta = torch.tensor(datapkl['betas'], dtype=torch.float)
        theta = torch.tensor(datapkl['pose'], dtype=torch.float)

        return img, segmask, theta, beta

    def get_name(self, idx):
        return self.img_names[idx]
