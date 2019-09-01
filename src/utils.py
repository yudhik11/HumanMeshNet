import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as TVF
import torch

def get_joints_from_surf(verts, smpl):
    joint_x = torch.matmul(verts[:, :, 0], smpl.J_regressor)
    joint_y = torch.matmul(verts[:, :, 1], smpl.J_regressor)
    joint_z = torch.matmul(verts[:, :, 2], smpl.J_regressor)

    joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)
    return joints

def SMPLJ_to_H36MJ16(joints, verts):
    SMPL_to_H36M_indices16 = torch.tensor([11, 5, 2, 1, 4, 10, 0, 6, 12, 12, 21, 19, 17,
                                         16, 18, 20]).to(verts.device)
    head_index = 410

    new_joints = joints[:, SMPL_to_H36M_indices16, :]
    new_joints[:, 9, :] = verts[:, head_index, :]
    return new_joints

def SMPLJ_to_H36MJ14(joints, verts):
    SMPL_to_H36M_indices14 = torch.tensor([11, 5, 2, 1, 4, 10, 12, 12, 21, 19, 17,
                                         16, 18, 20]).to(verts.device)
    head_index = 410

    new_joints = joints[:, SMPL_to_H36M_indices14, :]
    new_joints[:, 7, :] = verts[:, head_index, :]
    return new_joints

def flip_smpl(theta):
    theta = theta.reshape(-1, 3)
    indices = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    theta = theta[indices]
    theta[:, 1:3] = theta[:, 1:3]*-1
    return theta.reshape(-1)


def get_num_correct_class(pred, gt):
    """
    Get correct number of predictions from predicted class affinities and groundtruth labels
    """

    return (pred.argmax(1) == gt).sum()


def read_verts(inp_mesh_path):
    verts = []
    with open(inp_mesh_path, "r") as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split()
        if l[0] == "v":
            verts.append(list(map(float, l[1:4])))

    verts = np.array(verts)
    verts = torch.tensor(verts)
    return verts

def show(img):
    """
    Show torch image
    """

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def show1C(img):
    """
    Show 1C image
    """
    plt.imshow(img, interpolation='nearest')



def smart_padding(img):
    """
    Smart padding of PIL image to pad it in minimal way to make it square
    Input : H*W*3
    Output : K*K*3 where K=max(H,W)
    """
    desired_size = max(img.size[0], img.size[1])

    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width //2
    pad_height = delta_height //2
    return TVF.pad(img, (pad_width, pad_height, delta_width-pad_width, delta_height-pad_height))


def smart_padding_iuv(img):
    """
    Smart padding of numpy image to pad it in minimal way to make it square
    Input : H*W*3
    Output : K*K*3 where K=max(H,W)
    """
    desired_size = max(img.shape[0], img.shape[1])

    delta_width = desired_size - img.shape[0]
    delta_height = desired_size - img.shape[1]
    pad_width = delta_width //2
    pad_height = delta_height //2
    return np.pad(img, [(0, 0), (pad_width, delta_width-pad_width), (pad_height, delta_height-pad_height)], 'constant')



def smart_padding_depth(img):
    """
    Smart padding of numpy depth image to pad it in minimal way to make it square
    Input : H*W
    Output : K*K where K=max(H,W)
    """
    desired_size = max(img.shape[0], img.shape[1])

    delta_width = desired_size - img.shape[0]
    delta_height = desired_size - img.shape[1]
    pad_width = delta_width //2
    pad_height = delta_height //2

    return np.pad(img, [(pad_width, delta_width-pad_width), (pad_height, delta_height-pad_height)], 'constant')


def get_regularization_matrix(smpl):
    N = 6890
    regu = np.zeros((6890, 6890))
    for a, b, c in smpl.faces.cpu().numpy().astype(np.long):
        regu[a, b] = 1
        regu[a, c] = 1
        regu[b, a] = 1
        regu[b, c] = 1
        regu[c, a] = 1
        regu[c, b] = 1

    degree = regu.sum(1)[:, np.newaxis]
    final_regu = regu / degree

    return torch.tensor(final_regu.T, dtype=torch.float)

def regularize_mesh(surf, regu):
    inp_shape = surf.shape
    surf = surf.view(-1, 6890, 3)
    surf_x = torch.matmul(surf[:, :, 0], regu).unsqueeze(2)
    surf_y = torch.matmul(surf[:, :, 1], regu).unsqueeze(2)
    surf_z = torch.matmul(surf[:, :, 2], regu).unsqueeze(2)

    final_surf = torch.cat((surf_x, surf_y, surf_z), dim=2)
    return final_surf.view(inp_shape)

def orthographic_projection(X, camera, dataset='surreal'):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X = X.view(-1, 24, 3)
    if dataset == 'surreal':
        X_trans = torch.zeros_like(X[:, :, 1:])
        X_trans[:, :, 0] = X[:, :, 2]
        X_trans[:, :, 1] = X[:, :, 1]
        X_trans += camera[:, :, 1:]
    else:
        X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


def visualise_keypoints(gt_keypoints2d, img=None):
    if img is None:
        plt.scatter(list(gt_keypoints2d[:, 0].cpu().numpy()) , list(gt_keypoints2d[:, 1].cpu().numpy()))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    else:
        gt_keypoints2d = gt_keypoints2d.clone()
        gt_keypoints2d*=224
        npimg = img.numpy()
        gt_keypoints2d = gt_keypoints2d.view(24, 2)
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.plot(list(gt_keypoints2d[:, 0]),list(gt_keypoints2d[:, 1]),'o')
        plt.show()
