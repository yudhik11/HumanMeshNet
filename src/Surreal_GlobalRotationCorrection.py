import transforms3d
import math
import numpy as np
import scipy.io as sio

# Returns intrinsic camera matrix
# Parameters are hard-coded since all SURREAL images use the same.
def get_intrinsic():
    # These are set in Blender (datageneration/main_part1.py)
    res_x_px = 320  # *scn.render.resolution_x
    res_y_px = 240  # *scn.render.resolution_y
    f_mm = 60  # *cam_ob.data.lens
    sensor_w_mm = 32  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

    scale = 1  # *scn.render.resolution_percentage/100
    skew = 0  # only use rectangular pixels
    pixel_aspect_ratio = 1

    # From similar triangles:
    # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    # Center of the image
    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    # Intrinsic camera matrix
    K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
    return K


# Returns extrinsic camera matrix
#   T : translation vector from Blender (*cam_ob.location)
#   RT: extrinsic computer vision camera matrix
#   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv


def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec


# provide info path
# we need camLoc and zrot from it to correct global rotation
info_path = \
"/media/HDD_2TB/sourabh/surreal_complete/surreal/download/dump/SURREAL/data/cmu/train/run0/01_01/01_01_c0001_info.mat"
fno = 0 # frame number

info = sio.loadmat(info_path)
zrot = info['zrot'][0][0]
RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                    (math.sin(zrot), math.cos(zrot), 0),
                    (0, 0, 1)))

intrinsic = get_intrinsic()
extrinsic, R, T = get_extrinsic(info['camLoc'])

# change global rotation to align wrt camera
pose = info['pose'][:, fno]
pose[0:3] = rotateBody(RzBody, pose[0:3])

shape = info['shape'][:, fno]