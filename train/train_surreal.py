import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18

import torch.nn.functional as F
import torch.nn as nn

import logging
import os
import sys
import numpy as np

sys.path.append(os.path.abspath("../"))

from src import utils

from src.models import MyResnet18, MyFCNet2
from src.Surreal_dataset import SurrealBothSMRGBCrop_FlipCorrect
from src.eval_metric import procrustes_hmr, MPJPE
from src.SMPL_pytorch import SMPL

from matplotlib import pyplot as plt

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

num_workers = 6
batch_size = 64
learning_rate = 0.0001

save_model_path = "../../logs/DPSMRGB_Surf/"
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
logfile_path = "../../logs/DPSMRGB_Surf.log"

log_freq = 500
validate_freq = 1000
save_freq = 3000



logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
logging.info(torch.__version__)
logging.info(device)
logging.info("------------------------------------------------------------------------")



class MyDRNetwork(torch.nn.Module):
    def __init__(self):
        super(MyDRNetwork, self).__init__()

        self.SM_cnn_s = resnet18(pretrained=True)
        self.SM_cnn_s.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.RGB_cnn_s = resnet18(pretrained=True)

        self.R_fc_surface = MyFCNet2(2*1000, 6890*3)
        self.R_fc_joints = MyFCNet2(2*1000, 24*3)

    def forward(self, rgbs, segs):
        """
        with torch.no_grad():
            prob = self.C_net(x)
            prob = F.softmax(prob, dim=1)
            idxs = prob.argmax(dim=1)
            prior = self.cluster_centers[idxs].to(prob.device)
        """

        feat_segs_s = self.SM_cnn_s(segs)
        feat_rgbs_s = self.RGB_cnn_s(rgbs)
        y = torch.cat((feat_segs_s, feat_rgbs_s), dim=1)
        surf = self.R_fc_surface(y)
        joints = self.R_fc_joints(y)

        return joints, surf

train_dataset = SurrealBothSMRGBCrop_FlipCorrect("train")
val_dataset = SurrealBothSMRGBCrop_FlipCorrect("val")

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)


model = MyDRNetwork().to(device)
smpl = SMPL().to(device)

criterion1 = torch.nn.MSELoss().to(device)
criterion2 = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

reg_mat = utils.get_regularization_matrix(smpl=smpl).to(device)


start_epoch = 0
num_epochs = 50


def validate():
    model.eval()
    val_loss = 0.0
    mpjpe_j = 0.0
    mpjpe_j_pa = 0.0
    mpjpe_s = 0.0
    mpjpe_s_pa = 0.0
    mpjpe_js = 0.0
    mpjpe_js_pa = 0.0

    with torch.no_grad():
        for i, (imgs, _, dpsegs, gt_theta, gt_beta, gt_joints) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            dpsegs = dpsegs.to(device)
            gt_theta = gt_theta.to(device)
            gt_beta = gt_beta.to(device)

            #zero_shape = torch.zeros(imgs.size(0), 10).to(device)
            gt_S, gt_J, _ = smpl(gt_beta, gt_theta, get_skin=True)

            out_J, out_S = model(imgs, dpsegs)
            out_S = utils.regularize_mesh(regu=reg_mat, surf=out_S)

            out_S = out_S.view(gt_S.shape)
            out_J = out_J.view(gt_J.shape)

            out_JS = utils.get_joints_from_surf(out_S, smpl)

            loss = criterion1(out_S, gt_S) + 100*criterion1(out_J, gt_J)

            val_loss += loss.item()

            out_J = out_J.cpu().numpy()
            gt_J = gt_J.cpu().numpy()

            out_S = out_S.cpu().numpy()
            gt_S = gt_S.cpu().numpy()
            out_JS = out_JS.cpu().numpy()

            for j in range(imgs.size(0)):
                mpjpe_j += MPJPE(out_J[j], gt_J[j])
                oj, gj = procrustes_hmr(out_J[j], gt_J[j])
                mpjpe_j_pa += MPJPE(oj, gj)

                mpjpe_s += MPJPE(out_S[j], gt_S[j])
                os, gs = procrustes_hmr(out_S[j], gt_S[j])
                mpjpe_s_pa += MPJPE(os, gs)

                mpjpe_js += MPJPE(out_JS[j], gt_J[j])
                oj, gj = procrustes_hmr(out_JS[j], gt_J[j])
                mpjpe_js_pa += MPJPE(oj, gj)


    logging.info("Validation Loss : {:0.6f} | MPJPE_J : {:0.6f} | MPJPE_J_PA : {:0.6f} \
    | MPJPE_S : {:0.6f} | MPJPE_S_PA : {:0.6f} | MPJPE_JS : {:0.6f} | MPJPE_JS_PA : {:0.6f}".format(
                val_loss / len(val_dataloader),
                mpjpe_j / len(val_dataset),
                mpjpe_j_pa / len(val_dataset),
                mpjpe_s / len(val_dataset),
                mpjpe_s_pa / len(val_dataset),
                mpjpe_js / len(val_dataset),
                mpjpe_js_pa / len(val_dataset),
    ))

    model.train()


def train(epoch):
    model.train()
    total_iters = len(train_dataloader)

    total_loss = 0.0
    running_loss = 0.0

    for i, (imgs, _, dpsegs, gt_theta, gt_beta, gt_joints) in enumerate(train_dataloader):
        imgs = imgs.to(device)
        dpsegs = dpsegs.to(device)
        gt_theta = gt_theta.to(device)
        gt_beta = gt_beta.to(device)

        #zero_shape = torch.zeros(imgs.size(0), 10).to(device)
        gt_S, gt_J, _ = smpl(gt_beta, gt_theta, get_skin=True)

        out_J, out_S = model(imgs, dpsegs)
        out_S = utils.regularize_mesh(surf=out_S, regu=reg_mat)

        out_S = out_S.view(gt_S.shape)
        out_J = out_J.view(gt_J.shape)

        out_JS = utils.get_joints_from_surf(out_S, smpl)

        optimizer.zero_grad()

        loss = criterion1(out_S, gt_S) + 100*criterion1(out_J, gt_J)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        iters = epoch * total_iters + i + 1
        if iters % log_freq == 0:
            logging.info("Epoch {:02d} [{:05d}/{:05d}] Loss : {:.6f}".format(
                    epoch, i, total_iters, running_loss/log_freq
            ))
            running_loss = 0.0
        if iters % validate_freq == 0:
            validate()
        if iters % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_model_path, 'e{}-i{}.ckpt'.format(epoch,i)))

    logging.info("Epoch {} Finished Training Loss : {:0.6f}".format(epoch, total_loss))
    validate()
    torch.save(model.state_dict(), os.path.join(save_model_path, 'e{}-i{}.ckpt'.format(epoch,i)))

for epoch in range(start_epoch, num_epochs):
    logging.info("Epoch {} started".format(epoch))
    train(epoch)
