
'''
    file:   human36m_dataloader.py

    author: zhangxiong(1025679612@qq.com)
    date:   2018_05_09
    purpose:  load hum3.6m data
'''

import sys
from torch.utils.data import Dataset, DataLoader
import os 
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

# sys.path.append('./src')
from .utils import calc_aabb, cut_image, flip_image, draw_lsp_14kp__bone, rectangle_intersect, get_rectangle_intersect_ratio, convert_image_by_pixformat_normalize, reflect_pose, reflect_lsp_kp
from .data_config import args
from .timer import Clock

class hum36m_dataloader(Dataset):
    def __init__(self, data_set_path, annotaion_path, use_crop, scale_range, use_flip, min_pts_required, pix_format = 'NHWC', normalize = False, flip_prob = 0.3):
        self.data_folder = data_set_path
        self.anno_file_path = annotaion_path
        self.use_crop = use_crop
        self.scale_range = scale_range
        self.use_flip = use_flip
        self.flip_prob = flip_prob
        self.min_pts_required = min_pts_required
        self.pix_format = pix_format
        # self.normalize = normalize # usually false
        self.normalize = True # usually false
        self._load_data_set()

    def _load_data_set(self):
        
        clk = Clock()

        self.images = []
        self.kp2ds  = []
        self.boxs   = []
        self.kp3ds  = []
        self.shapes = []
        self.poses  = []

        print('start loading hum3.6m data.')
        # import pdb;pdb.set_trace()
        anno_file_path = os.path.join(self.data_folder,self.anno_file_path)
        # anno_file_path = "/annotation_body3d/fps10/h36m_test.npz"
        # import pdb;pdb.set_trace()

        # anno_file_path = os.path.join(self.data_folder, 'cameras.h5')
        # with h5py.File(anno_file_path) as fp:
        with np.load(anno_file_path, allow_pickle=True) as fp:
            # import pdb;pdb.set_trace()
            total_center = np.array(fp['center']) #total_pose
            total_kp2d = np.array(fp['part'])
            total_kp3d = np.array(fp['S'])
            total_scale = np.array(fp['scale']) #total_shape
            total_image_names = np.array(fp['imgname'])

            # import pdb;pdb.set_trace()
            assert len(total_kp2d) == len(total_kp3d) and len(total_kp2d) == len(total_image_names) and \
                len(total_kp2d) == len(total_scale) and len(total_kp2d) == len(total_center)

            # l =  110232
            l = len(total_kp2d)
            def _collect_valid_pts(pts):
                r = []
                for pt in pts:
                    # if pt[2] != 0:
                    if pt[1] != 0:
                        r.append(pt)
                return r

            for index in range(l):
                # kp2d = total_kp2d[index].reshape((-1, 3))
                kp2d = total_kp2d[index].reshape((-1, 2))
                # import pdb;pdb.set_trace()
                # if np.sum(kp2d[:, 2]) < self.min_pts_required: # chnaged to make the code work
                    # continue
                if np.sum(kp2d[:, 1]) < self.min_pts_required:
                    continue

                lt, rb, v = calc_aabb(_collect_valid_pts(kp2d))
                self.kp2ds.append(np.array(kp2d.copy(), dtype = np.float))
                self.boxs.append((lt, rb))
                # import pdb;pdb.set_trace()
                # self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.kp3ds.append(total_kp3d[index].copy().reshape(-1, 3))
                self.shapes.append(total_scale[index].copy())
                self.poses.append(total_center[index].copy())
                # self.images.append(os.path.join(self.data_folder, 'image') + total_image_names[index].decode())
                self.images.append(os.path.join(self.data_folder, 'images/') + total_image_names[index])

        print('finished load hum3.6m data, total {} samples'.format(len(self.kp3ds)))
        
        clk.stop()

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_path = self.images[index]
        kps = self.kp2ds[index].copy()
        box = self.boxs[index]
        kp_3d = self.kp3ds[index].copy()
        shapes = self.shapes[index]
        center = self.poses[index]

        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        # scale = [1,1,1,1] # run this?
        # import pdb;pdb.set_trace()
        image, kps, leftTop = cut_image(image_path, kps, scale, box[0], box[1])
        # image = cv2.imread(image_path) #use when not cropping
#use 121 and 122 when using cropping(the next 2 lines)
        ratio = 1.0 * args.crop_size / image.shape[0] #args.crop_size = 224 for resnet
        kps[:, :2] *= ratio
        dst_image = cv2.resize(image, (args.crop_size, args.crop_size), interpolation = cv2.INTER_CUBIC)

        trival, shape, pose = np.zeros(3), self.shapes[index], self.poses[index]

        if self.use_flip and random.random() <= self.flip_prob:
            dst_image, kps = flip_image(dst_image, kps)
            pose = reflect_pose(pose)
            kp_3d = reflect_lsp_kp(kp_3d)

        #normalize kp to [-1, 1]
        #no normalize to check
        ratio2 = 1.0 / args.crop_size
        # ratio2 = 1.0 / 1000  # ratio according to image shape 12/2/2020
        kps[:, :2] = 2.0 * kps[:, :2] * ratio2 - 1.0

        # normalize kp_3d to [-1,1] change added to see how it effects training
        # min_values =  np.min(kp_3d[:, :3], axis=0)
        # max_values =  np.max(kp_3d[:, :3], axis=0)

        # kp_3d[:, :3] = (kp_3d[:, :3] - min_values) * 2 / (max_values - min_values + 1e-7) -1 # add small 1e-7
        kp_3d[1:] -= kp_3d[:1]

        # theta = np.concatenate((trival, pose, shape), axis = 0)

        return {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'kp_2d': torch.from_numpy(kps).float(),
            'kp_3d': torch.from_numpy(kp_3d).float(),
            # 'theta': torch.from_numpy(theta).float(),
            'left_top' : torch.FloatTensor(leftTop),
            'image_name': self.images[index],
            'w_smpl':1.0,
            'w_3d':1.0,
            'data_set':'hum3.6m',
            'ratio':ratio,
            'ratio2': ratio2,
            'shapes' : shapes,
            'center' : center

        }

def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    #shape of kps_mask (1002, 1000, 3)
    # import pdb;pdb.set_trace()
    # Draw the keypoints.
    for l in range(len(kps_lines)):
        # import pdb;pdb.set_trace()
        i1 = kps_lines[l][0] #0 for l = 0
        i2 = kps_lines[l][1] #7 for l = 0 check skeliton shape tupple   kps shape [3,17]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            # import pdb;pdb.set_trace()
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def world2cam(world_coord, R, t):
    print(f'word coordinate shape :{world_coord.shape}')
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c): #cam_coord shape (17,3)
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    print(f'image_coord shape {img_coord.shape}')
    return img_coord

if __name__ == '__main__':
    # h36m = hum36m_dataloader('/netscratch/nafis/human-pose/human36_dataset', True, [1.1, 2.0], True, 5, flip_prob = 1)
    print("start")
    import matplotlib.pyplot as plt

    annotation_path_train = "annotation_body3d/fps50/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps50/h36m_test.npz"
    root_data_path = '/netscratch/nafis/human-pose/dataset_hum36m_f50'
    h36m = hum36m_dataloader(root_data_path,annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
    l = len(h36m)
    filename = '/netscratch/nafis/human-pose/pytorch_HMR/src/results/test'
    for _ in range(l):
        r = h36m.__getitem__(_)
        # import pdb;pdb.set_trace()
        print(r['kp_2d'].shape)
        cvimg = cv2.imread(r['image_name'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # tmpkps = np.zeros((3,17))
        # tmpkps = np.zeros((3,17))
        # tmpkps[0,:], tmpkps[1,:] = r['kp_2d'][:,0], r['kp_2d'][:,1]
        # tmpkps[2,:] = 1
        # tmpkps = np.zeros((17,3))
        tmpkps = r['kp_3d']
        # tmpkps[:,0], tmpkps[:,1] = r['kp_3d'][:,0], r['kp_3d'][:,1]
        # tmpkps[:,2] = 1
        # print(tmpkps.shape) # (17,3)
        skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        #convert key points to pixel coordinate before visualize
        R = np.array([[-0.91536173,  0.05154812, -0.39931903],
                    [ 0.40180837,  0.18037357, -0.89778361],
                    [ 0.02574754, -0.98224649, -0.18581953]])
        f = np.array([[1145.04940459],[1143.78109572]])
        c = np.array([[512.54150496],[515.45148698]])
        T = np.array([[1.84110703],
                    [4.95528462],
                    [1.5634454 ]])
        k = np.array([[-0.20709891],
                    [ 0.24777518],
                    [-0.00307515]])
        print("#########")
        # tmpkps = world2cam(tmpkps, R, T)
        # print(tmpkps)
        tmpkps = cam2pixel(tmpkps, f, c) # only in cam coordinate
        # print(tmpkps.shape)  tmpkps shape  (17,3)
        tmpimg = vis_keypoints(cvimg, tmpkps.transpose(1,0), skeleton)
        cv2.imwrite(filename + '_output_kps3d_origi_scale_1000.jpg', tmpimg)
        break