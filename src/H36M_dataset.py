import torch
import torch.utils.data as data

import torchvision.transforms as transforms

import scipy.io
from PIL import Image
import os

import numpy as np
import cv2

from .utils import smart_padding, smart_padding_depth, smart_padding_iuv


class H36MDataset_S9_seq(data.Dataset):
    def __init__(self, prefix):
        super(H36MDataset_S9_seq, self).__init__()
                        
        self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train/"
        LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_gt3ds.mat"
        theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_pose.mat"
        beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_shape.mat"
        # dense pose dir
        self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/s9_train_segm"
        self.iuv_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/s9_train_IUV"
        # invalid denspose output list
        invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/s9_train.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        
        self.pose_images = sorted(os.listdir(self.root_dir))
        #self.length = len(self.pose_images)
        
        self.seq_idxs = [i for i, pm in enumerate(self.pose_images) if pm.startswith(prefix)]
        self.length = len(self.seq_idxs)
        print("Length:", self.length)
                
        self.landmarks = torch.tensor(scipy.io.loadmat(LM_matfile)['gt3ds'], dtype=torch.float)
        self.thetas = torch.tensor(scipy.io.loadmat(theta_matfile)['poses'], dtype=torch.float)
        self.betas = torch.tensor(scipy.io.loadmat(beta_matfile)['shapes'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def get_name(self, idx):
        return self.pose_images[idx]
    
    def __getitem__(self, idx):
        idx = self.seq_idxs[idx]
        
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = idx-1
            img_name = self.pose_images[idx]
        
        seg_name = img_name[:-4] + "_IUV.mat"
            
        seg_path = os.path.join(self.dp_root_dir, seg_name)
        seg = scipy.io.loadmat(seg_path)['segm']
        seg = smart_padding_depth(seg) 
        seg = cv2.resize(seg, (224, 224)) # resize to 224
        seg = torch.tensor(seg).unsqueeze(0).float()
        
        iuv_path = os.path.join(self.iuv_root_dir, seg_name)
        iuv = scipy.io.loadmat(iuv_path)['segm']
        iuv = smart_padding_iuv(iuv) 
        iuv = cv2.resize(iuv, (224, 224)) # resize to 224
        iuv = np.transpose(iuv, (2, 0, 1))
        iuv = torch.tensor(iuv).unsqueeze(0).float()
        
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.tsfm(img)        
        
        return img, seg, iuv, self.thetas[idx], self.betas[idx], self.landmarks[idx].view(-1, 3)


class H36MDatasetFull(data.Dataset):
    def __init__(self, dstype):
        super(H36MDatasetFull, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/total_images_train/"
            LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/total_train_gt3ds.mat"
            theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/total_train_pose.mat"
            beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/total_train_shape.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/total_train/"
            
        elif dstype == "val":
            self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/"
            LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/"
            theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/"
            beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/"
                        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        
        if dstype == "val":
            self.pose_images = sorted(os.listdir(os.path.join(self.root_dir, "images_train") ))
            self.pose_images = [os.path.join("images_train", pim) for pim in self.pose_images]
            self.ids = [0]*len(self.pose_images)
            self.pose_images.extend( [os.path.join("images_val", pim) for pim in  \
                sorted(os.listdir(os.path.join(self.root_dir, "images_val") )) ] )
            self.ids.extend( [1]*(len(self.pose_images) - len(self.ids)) )
            
        elif dstype == "train":
            self.pose_images = sorted(os.listdir(self.root_dir))
            
        self.length = len(self.pose_images)
        
        self.dstype = dstype
        if dstype == "train":
            self.landmarks = torch.tensor(scipy.io.loadmat(LM_matfile)['gt3ds'], dtype=torch.float)
            self.thetas = torch.tensor(scipy.io.loadmat(theta_matfile)['poses'], dtype=torch.float)
            self.betas = torch.tensor(scipy.io.loadmat(beta_matfile)['shapes'], dtype=torch.float)
        elif dstype == "val":
            self.landmarks1 = torch.tensor(scipy.io.loadmat(os.path.join(LM_matfile, "train_gt3ds.mat") )['gt3ds'], dtype=torch.float)
            self.thetas1 = torch.tensor(scipy.io.loadmat(os.path.join(theta_matfile, "train_pose.mat"))['poses'], dtype=torch.float)
            self.betas1 = torch.tensor(scipy.io.loadmat(os.path.join(beta_matfile, "train_shape.mat"))['shapes'], dtype=torch.float)
            
            self.landmarks2 = torch.tensor(scipy.io.loadmat(os.path.join(LM_matfile, "val_gt3ds.mat") )['gt3ds'], dtype=torch.float)
            self.thetas2 = torch.tensor(scipy.io.loadmat(os.path.join(theta_matfile, "val_pose.mat"))['poses'], dtype=torch.float)
            self.betas2 = torch.tensor(scipy.io.loadmat(os.path.join(beta_matfile, "val_shape.mat"))['shapes'], dtype=torch.float)
            
            self.landmarks = torch.cat((self.landmarks1, self.landmarks2), dim=0)
            self.thetas = torch.cat((self.thetas1, self.thetas2), dim=0)
            self.betas = torch.cat((self.betas1, self.betas2), dim=0)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        # load IUV image
        img_name = self.pose_images[idx]
 
        if self.dstype == "train":
            seg_name = img_name[:-4] + "_IUV.mat"
        else:
            if self.ids[idx] == 0:
                seg_name = os.path.join("train", img_name.split("/")[-1][:-4] + "_IUV.mat")
            else:
                seg_name = os.path.join("val", img_name.split("/")[-1][:-4] + "_IUV.mat")   
        seg_path = os.path.join(self.dp_root_dir, seg_name)

        while not os.path.exists(seg_path):
            idx = np.random.randint(self.length)
            img_name = self.pose_images[idx]
            
            if self.dstype == "train":
                seg_name = img_name[:-4] + "_IUV.mat"
            else:
                if self.ids[idx] == 0:
                    seg_name = os.path.join("train", img_name.split("/")[-1][:-4] + "_IUV.mat")
                else:
                    seg_name = os.path.join("val", img_name.split("/")[-1][:-4] + "_IUV.mat")           
            seg_path = os.path.join(self.dp_root_dir, seg_name)
        
        seg = scipy.io.loadmat(seg_path)['segm']
        seg = smart_padding_depth(seg) 
        seg = cv2.resize(seg, (224, 224)) # resize to 224
        seg = torch.tensor(seg).unsqueeze(0).float()

        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.tsfm(img)        
        
        return img, seg, self.thetas[idx], self.betas[idx], self.landmarks[idx].view(-1, 3)
                

        
class H36MDataset_S9(data.Dataset):
    def __init__(self, dstype):
        super(H36MDataset_S9, self).__init__()
                        
        if dstype == "train" or dstype == "val":
            self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train/"
            LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_gt3ds.mat"
            theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_pose.mat"
            beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/s9_train_shape.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/s9_train_segm"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/s9_train.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.landmarks = torch.tensor(scipy.io.loadmat(LM_matfile)['gt3ds'], dtype=torch.float)
        self.thetas = torch.tensor(scipy.io.loadmat(theta_matfile)['poses'], dtype=torch.float)
        self.betas = torch.tensor(scipy.io.loadmat(beta_matfile)['shapes'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def get_name(self, idx):
        return self.pose_images[idx]
    
    def __getitem__(self, idx):
        
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        
        seg_name = img_name[:-4] + "_IUV.mat"
            
        seg_path = os.path.join(self.dp_root_dir, seg_name)
        seg = scipy.io.loadmat(seg_path)['segm']
        seg = smart_padding_depth(seg) 
        seg = cv2.resize(seg, (224, 224)) # resize to 224
        seg = torch.tensor(seg).unsqueeze(0).float()

        
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.tsfm(img)        
        
        return img, seg, self.thetas[idx], self.betas[idx], self.landmarks[idx].view(-1, 3)

        
        
class H36MDatasetTest(data.Dataset):
    def __init__(self, dstype):
        super(H36MDatasetTest, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/images_train"
            LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/train_gt3ds.mat"
            theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/train_pose.mat"
            beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/train_shape.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/train/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/train.txt"
            
        elif dstype == "val":
            self.root_dir = "/home/abbhinav/h36_smpl/hmr/src/datasets/images_val"
            LM_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/val_gt3ds.mat"
            theta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/val_pose.mat"
            beta_matfile = "/home/abbhinav/h36_smpl/hmr/src/datasets/val_shape.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/val/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m_smpl/val.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.landmarks = torch.tensor(scipy.io.loadmat(LM_matfile)['gt3ds'], dtype=torch.float)
        self.thetas = torch.tensor(scipy.io.loadmat(theta_matfile)['poses'], dtype=torch.float)
        self.betas = torch.tensor(scipy.io.loadmat(beta_matfile)['shapes'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def get_name(self, idx):
        return self.pose_images[idx]
    
    def __getitem__(self, idx):
        
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        
        seg_name = img_name[:-4] + "_IUV.mat"
            
        seg_path = os.path.join(self.dp_root_dir, seg_name)
        seg = scipy.io.loadmat(seg_path)['segm']
        seg = smart_padding_depth(seg) 
        seg = cv2.resize(seg, (224, 224)) # resize to 224
        seg = torch.tensor(seg).unsqueeze(0).float()

        
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.tsfm(img)        
        
        return img, seg, self.thetas[idx], self.betas[idx], self.landmarks[idx].view(-1, 3)
        


class H36MDataset_DPSMRGB(data.Dataset):
    def __init__(self, dstype):
        super(H36MDataset_DPSMRGB, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/train"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_train_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train.txt"
            
        elif dstype == "val":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/test"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_test_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        """
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = scipy.io.loadmat(img_path)['tmp']
        """
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.tsfm(img).float()
        
        segs_name = img_name[:-4] + "_IUV.mat"   
        segs_path = os.path.join(self.dp_root_dir, segs_name)
        segs = scipy.io.loadmat(segs_path)['segm']
        segs = smart_padding_depth(segs) # smart pad to make square
        segs = cv2.resize(segs, (224, 224)) # resize to 224
        segs = torch.tensor(segs).unsqueeze(0).float()
        
        joints = self.joints[idx].view(-1, 3) / 1000.0
        joints = joints - joints[6].unsqueeze(0)

        return img, segs, joints



class H36MDataset(data.Dataset):
    def __init__(self, dstype):
        super(H36MDataset, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/train"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_train_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train.txt"
            
        elif dstype == "val":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/test"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_test_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding_depth(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        """
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = scipy.io.loadmat(img_path)['tmp']
        """
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        img_name = img_name[:-4] + "_IUV.mat"
            
        img_path = os.path.join(self.dp_root_dir, img_name)
        img = scipy.io.loadmat(img_path)['segm']
        
        img = smart_padding_depth(img) # smart pad to make square
        img = cv2.resize(img, (224, 224)) # resize to 224
        img = torch.tensor(img).unsqueeze(0)
        
        joints = self.joints[idx].view(-1, 3) / 1000.0
        joints = joints - joints[6].unsqueeze(0)
        
        return img.float(), joints

    
    
class H36MDataset_4C(data.Dataset):
    def __init__(self, dstype):
        super(H36MDataset_4C, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/train"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_train_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train.txt"
            
        elif dstype == "val":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/test"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_test_joints.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'], dtype=torch.float)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        """
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = scipy.io.loadmat(img_path)['tmp']
        """
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        
        ori_img_name = img_name
        ori_img_path = os.path.join(self.root_dir, ori_img_name)
        ori_img = Image.open(ori_img_path).convert('RGB')
        ori_img = self.tsfm(ori_img).float()
        
        img_name = img_name[:-4] + "_IUV.mat"   
        img_path = os.path.join(self.dp_root_dir, img_name)
        img = scipy.io.loadmat(img_path)['segm']
        img = smart_padding_depth(img) # smart pad to make square
        img = cv2.resize(img, (224, 224)) # resize to 224
        img = torch.tensor(img).unsqueeze(0).float()
        
        combined_img = torch.cat((ori_img, img), dim=0)

        joints = self.joints[idx].view(-1, 3) / 1000.0
        joints = joints - joints[6].unsqueeze(0)

        return combined_img, joints

    

class H36MDatasetCJ(data.Dataset):
    def __init__(self, dstype):
        super(H36MDatasetCJ, self).__init__()
                        
        if dstype == "train":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/train"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_train_joints.mat"
            clustermatfile_j = "/media/HDD_2TB/himansh/new_human3.6m/part_clusters/full_body/cluster_train_500.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/train.txt"
            
        elif dstype == "val":
            self.root_dir = "/media/HDD_2TB/himansh/new_human3.6m/test"
            jointsmatfile = "/media/HDD_2TB/himansh/new_human3.6m/new_test_joints.mat"
            clustermatfile_j = "/media/HDD_2TB/himansh/new_human3.6m/part_clusters/full_body/cluster_test_500.mat"
            # dense pose dir
            self.dp_root_dir = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test/"
            # invalid denspose output list
            invalid_path = "/home/saketh/Densepose/densepose/DensePoseData/h36m/test.txt"
                
        # get invalid image names
        with open(invalid_path) as f:
            self.invalid_images = f.readlines()
        self.invalid_images = [ii.strip() for ii in self.invalid_images]
        
        self.tsfm = transforms.Compose([
            transforms.Lambda(lambda img : smart_padding_depth(img)),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
            
        self.pose_images = sorted(os.listdir(self.root_dir))
        self.length = len(self.pose_images)
                
        self.joints = torch.tensor(scipy.io.loadmat(jointsmatfile)['joints'], dtype=torch.float)
        
        dct = scipy.io.loadmat(clustermatfile_j)
        
        if dstype == "train":
            self.cluster_labels_j = torch.tensor(dct['idx_joint'].astype(np.int), dtype=torch.long).squeeze() - 1   
        elif dstype == "val":
            self.cluster_labels_j = torch.tensor(dct['idx_centers'].astype(np.int), dtype=torch.long).squeeze() - 1   
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        """
        img_name = self.pose_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = scipy.io.loadmat(img_path)['tmp']
        """
        # load IUV image
        img_name = self.pose_images[idx]
        if img_name in self.invalid_images:
            idx = 0
            img_name = self.pose_images[idx]
        img_name = img_name[:-4] + "_IUV.mat"
            
        img_path = os.path.join(self.dp_root_dir, img_name)
        img = scipy.io.loadmat(img_path)['segm']
        
        img = smart_padding_depth(img) # smart pad to make square
        img = cv2.resize(img, (224, 224)) # resize to 224
        img = torch.tensor(img).unsqueeze(0)
        
        joints = self.joints[idx].view(-1, 3) / 1000.0
        joints = joints - joints[6].unsqueeze(0)
        
        return img.float(), joints, self.cluster_labels_j[idx]

