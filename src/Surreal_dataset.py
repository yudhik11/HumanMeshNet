import torch
import torch.utils.data as data

import torchvision.transforms as transforms

import scipy.io
from PIL import Image
import os

import numpy as np
import cv2

from .utils import smart_padding, smart_padding_depth, flip_smpl

"""
Dataloaders for direct surface regression
"""


class SurrealBothSMRGBCrop_FlipCorrect(data.Dataset):
    def __init__(self, dstype):
        super(SurrealBothSMRGBCrop_FlipCorrect, self).__init__()

        if dstype == "train":
            self.gtsm_root_dir = \
            "/media/SSD_150/abbhinav/body3/data/segm_train_run0/"
            posematfile = "/media/SSD_150/abbhinav/body3/data/poses_gr/train_pose_matrix.mat"
            shapematfile = \
            "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/shapes/train_shape_matrix.mat"
            jointsmatfile = "/media/SSD_150/abbhinav/body3/data/joints_gr/train_joint_matrix.mat"
            self.rgb_root_dir = "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/images_train_run0"
            # dense pose dir
            self.dp_root_dir = \
            "/home/saketh/Densepose/densepose/DensePoseData/surreal/train"
            # invalid denspose output list
            invalid_path = \
            "/home/saketh/Densepose/densepose/DensePoseData/surreal/train.txt"


        elif dstype == "val":
            self.gtsm_root_dir = \
            "/media/SSD_150/abbhinav/body3/data/segm_val_run0/"
            posematfile = "/media/SSD_150/abbhinav/body3/data/poses_gr/val_pose_matrix.mat"
            shapematfile = \
            "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/shapes/val_shape_matrix.mat"
            jointsmatfile = "/media/SSD_150/abbhinav/body3/data/joints_gr/val_joint_matrix.mat"
            self.rgb_root_dir = "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/images_val_run0"
            # dense pose dir
            self.dp_root_dir = \
            "/home/saketh/Densepose/densepose/DensePoseData/surreal/val"
            # invalid denspose output list
            invalid_path = \
            "/home/saketh/Densepose/densepose/DensePoseData/surreal/val.txt"


        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        self.rgb_images = sorted(os.listdir(self.rgb_root_dir))

        # self.length = len(self.pose_images)
        self.length = len(self.rgb_images) // 5

        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'].T, dtype=torch.float)
        self.pose_params = torch.tensor(scipy.io.loadmat(posematfile)['poses'].T, dtype=torch.float)
        self.shape_params = torch.tensor(scipy.io.loadmat(shapematfile)['shapes'].T, dtype=torch.float)


        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip().split(".")[0]+".jpg" for ii in self.invalid_images]
        # store valid indices to sample in case of invalid
        subsampled_images = self.rgb_images[::5]
        self.valid_indices = [i for i, pi in enumerate(subsampled_images) if pi not in self.invalid_images]
        self.invalid_indices = [i for i, pi in enumerate(subsampled_images) if pi in self.invalid_images]

    def __len__(self):
        return self.length

    def _crop_tight(self, img, segs, dpsegs):
        nz = np.nonzero(segs)
        if len(nz[0]) == 0:
            return img, segs, dpsegs
        miny, maxy = nz[0].min(), nz[0].max()
        minx, maxx = nz[1].min(), nz[1].max()
        img = img.crop((minx, miny, maxx+1, maxy+1))
        segs = segs[miny:maxy+1, minx:maxx+1]
        dpsegs = dpsegs[miny:maxy+1, minx:maxx+1]
        return img, segs, dpsegs

    def __getitem__(self, idx):

        # if invalid then sample a random valid
        if idx in self.invalid_indices:
            past = idx
            idx = self.valid_indices[np.random.randint(len(self.valid_indices)) ]

        # get original index
        idx = idx*5

        # load seg image
        seg_name = self.rgb_images[idx][:-4] + ".mat"
        seg_path = os.path.join(self.gtsm_root_dir, seg_name)
        segs = scipy.io.loadmat(seg_path)['tmp']

        # load original image
        img_name = self.rgb_images[idx]
        img_path = os.path.join(self.rgb_root_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # load IUV image
        dpseg_name = self.rgb_images[idx][:-4] + "_IUV.mat"
        dpseg_path = os.path.join(self.dp_root_dir, dpseg_name)
        dpsegs = scipy.io.loadmat(dpseg_path)['segm']

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        segs = np.flip(segs, 1)
        dpsegs = np.flip(dpsegs, 1)

        img, segs, dpsegs = self._crop_tight(img, segs, dpsegs)

        segs = smart_padding_depth(segs) # smart pad to make square
        segs = cv2.resize(segs, (224, 224)) # resize to 224
        segs = torch.tensor(segs).unsqueeze(0).float()

        dpsegs = smart_padding_depth(dpsegs) # smart pad to make square
        dpsegs = cv2.resize(dpsegs, (224, 224)) # resize to 224
        dpsegs = torch.tensor(dpsegs).unsqueeze(0).float()


        img = self.tsfm(img).float()
        theta = flip_smpl(self.pose_params[idx])

        return img, segs, dpsegs, theta, self.shape_params[idx], self.joints[idx].view(-1, 3)


class SurrealGTSMRGBCrop_FlipCorrect(data.Dataset):
    def __init__(self, dstype):
        super(SurrealGTSMRGBCrop_FlipCorrect, self).__init__()

        if dstype == "train":
            self.gtsm_root_dir = \
            "/media/SSD_150/abbhinav/body3/data/segm_train_run0/"
            posematfile = "/media/SSD_150/abbhinav/body3/data/poses_gr/train_pose_matrix.mat"
            shapematfile = \
            "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/shapes/train_shape_matrix.mat"
            jointsmatfile = "/media/SSD_150/abbhinav/body3/data/joints_gr/train_joint_matrix.mat"
            self.rgb_root_dir = "/home/abbhinav/body3/data/images_train_run0"

        elif dstype == "val":
            self.gtsm_root_dir = \
            "/media/SSD_150/abbhinav/body3/data/segm_val_run0/"
            posematfile = "/media/SSD_150/abbhinav/body3/data/poses_gr/val_pose_matrix.mat"
            shapematfile = \
            "/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/shapes/val_shape_matrix.mat"
            jointsmatfile = "/media/SSD_150/abbhinav/body3/data/joints_gr/val_joint_matrix.mat"
            self.rgb_root_dir = "/home/abbhinav/body3/data/images_val_run0"

        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        self.rgb_images = sorted(os.listdir(self.rgb_root_dir))

        # self.length = len(self.pose_images)
        self.length = len(self.rgb_images) // 5

        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'].T, dtype=torch.float)
        self.pose_params = torch.tensor(scipy.io.loadmat(posematfile)['poses'].T, dtype=torch.float)
        self.shape_params = torch.tensor(scipy.io.loadmat(shapematfile)['shapes'].T, dtype=torch.float)

    def __len__(self):
        return self.length

    def _crop_tight(self, img, segs):
        nz = np.nonzero(segs)
        if len(nz[0]) == 0:
            return img, segs
        miny, maxy = nz[0].min(), nz[0].max()
        minx, maxx = nz[1].min(), nz[1].max()
        img = img.crop((minx, miny, maxx+1, maxy+1))
        segs = segs[miny:maxy+1, minx:maxx+1]
        return img, segs

    def __getitem__(self, idx):

        # get original index
        idx = idx*5

        # load seg image
        seg_name = self.rgb_images[idx][:-4] + ".mat"
        seg_path = os.path.join(self.gtsm_root_dir, seg_name)
        segs = scipy.io.loadmat(seg_path)['tmp']

        # load original image
        img_name = self.rgb_images[idx]
        img_path = os.path.join(self.rgb_root_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # flip image and segs
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        segs = np.flip(segs, 1)

        img, segs = self._crop_tight(img, segs)

        segs = smart_padding_depth(segs) # smart pad to make square
        segs = cv2.resize(segs, (224, 224)) # resize to 224
        segs = torch.tensor(segs).unsqueeze(0).float()

        img = self.tsfm(img).float()

        theta = flip_smpl(self.pose_params[idx])

        return img, segs, theta, self.shape_params[idx], self.joints[idx].view(-1, 3)


    def get_img_name(self, idx):
        return self.rgb_images[5*idx]
