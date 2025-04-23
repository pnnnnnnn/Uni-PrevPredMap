import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from mmcv.runner import get_dist_info
import torch.nn.functional as F
import sys
from shapely.geometry import box, LineString, Polygon
from projects.mmdet3d_plugin.maptr.modules.ops.diff_ras.polygon import SoftPolygon
import json


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts


def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    patch_z = pc_range[5] - pc_range[2]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    new_pts[..., 2:3] = pts[..., 2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h, patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts


class GlobalMapVec:
    def __init__(self, map_cfg):
        self.bev_h = map_cfg['bev_h']
        self.bev_w = map_cfg['bev_w']
        self.pc_range = map_cfg['pc_range']
        self.load_map_path = map_cfg['load_map_path']
        self.save_map_path = map_cfg['save_map_path']
        self.num_classes = map_cfg['num_classes']
        self.bev_patch_h = self.pc_range[4] - self.pc_range[1]
        self.bev_patch_w = self.pc_range[3] - self.pc_range[0]
        self.dataset = map_cfg['dataset']
        if self.dataset == 'nusc':
            self.city_list = ['singapore-onenorth', 'boston-seaport',
                              'singapore-queenstown', 'singapore-hollandvillage']
        elif self.dataset == 'av2':
            self.city_list = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
        elif self.dataset == 'ruqi':
            self.city_list = ['guangzhou']
        else:
            raise NotImplementedError("not supported dataset")
        self.iter_based_reset = map_cfg.get('iter_based_reset', -1)
        
        self.global_map_dict = {}
        self.map_status = None
        self.epoch_point = -2
    
    def load_map(self, device):
        with open(self.load_map_path, 'r') as f:
            global_map_dict = json.load(f)
        f.close()
        # print("self.global_map_dict['WDC'].keys(): ", self.global_map_dict['WDC'].keys())
        # sys.exit()
        self.global_map_dict = {}
        for city_name in self.city_list:
            self.global_map_dict[city_name] = {}
            for i in range(self.num_classes):
                self.global_map_dict[city_name][i] = []
                for vec in global_map_dict[city_name][str(i)]:
                    self.global_map_dict[city_name][i].append(torch.Tensor(vec))

    def check_map(self, device, epoch, status, iteration):
        # 起始初始化
        if self.map_status is None:
            self.epoch_point = epoch
            self.map_status = status
            if self.load_map_path is not None:
                self.load_map(device)
            else:
                self.create_map(device, status)
        # 切换train val状态初始化
        elif status != self.map_status:
            self.epoch_point = epoch
            self.map_status = status
            self.create_map(device, status)
        # 训练过程重置
        elif self.iter_based_reset:
            # print("iteration: ", iteration)
            if self.map_status == 'train' and iteration % self.iter_based_reset == 0:
                self.epoch_point = epoch
                self.map_status = status
                self.reset_map(iteration)
        else:
            if epoch != self.epoch_point:
                self.epoch_point = epoch
                self.map_status = status
                self.reset_map()

    def reset_map(self, iteration=-1):
        for city_name in self.city_list:
            self.global_map_dict[city_name] = {}
            print("reset map", city_name, "for epoch", self.epoch_point, "status", self.map_status, "for iter", iteration)
            for i in range(self.num_classes):
                self.global_map_dict[city_name][i] = []

    def create_map(self, device, status):
        for city_name in self.city_list:
            print("create map", city_name, status, "on", device, "for epoch", self.epoch_point)
            self.global_map_dict[city_name] = {}
            for i in range(self.num_classes):
                self.global_map_dict[city_name][i] = []
        self.map_status = status

    def update_map(self, city_name, trans, raster, status):
        trans = torch.from_numpy(trans).float()
        e2g_R, e2g_T = trans[:3, :3], trans[:3, -1].reshape(3, 1)
        for i in range(self.num_classes):
            for vec in raster[i]:
                vec_global = e2g_R @ vec.cpu().permute(1, 0) + e2g_T
                # print("vec_global: ", vec_global.permute(1, 0))
                self.global_map_dict[city_name][i].append(vec_global.permute(1, 0))
        # sys.exit()

    def get_map(self, city_name, trans, status, device):
        rasterized_results = torch.zeros(self.num_classes, self.bev_h, self.bev_w, device=device)
        trans = torch.from_numpy(trans).float()
        e2g_R, e2g_T = trans[:3, :3], trans[:3, -1].reshape(3, 1)
        g2e_R, g2e_T = e2g_R.T, - e2g_R.T @ e2g_T
        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        patch_ego = torch.Tensor([[xmin, ymin, 0], [xmin, ymax, 0], [xmax, ymax, 0], [xmax, ymin, 0], [xmin, ymin, 0]])
        patch_global = e2g_R @ patch_ego.permute(1, 0) + e2g_T
        # patch_global_xy = box(patch_global[0, 0], patch_global[1, 0], patch_global[0, 1], patch_global[1, 1])
        patch_global_xy = Polygon(patch_global.permute(1, 0)[..., :2].tolist())
        # print(patch_global_xy)
        # sys.exit()
        for i in range(self.num_classes):
            list_vec = []
            for vec in self.global_map_dict[city_name][i]:
                # print("vec: ", vec)
                vec_xy = LineString(vec[..., :2])
                if vec_xy.intersects(patch_global_xy):
                    vec_ego = g2e_R @ vec.permute(1, 0) + g2e_T
                    # print("vec_ego: ", vec_ego)
                    vec_ego_norm = normalize_3d_pts(vec_ego.permute(1, 0), self.pc_range)
                    # print("vec_ego_norm: ", vec_ego_norm)
                    # 确认可以不截断，栅格化会自动忽略边界外的矢量
                    vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
                    vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w
                    vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h
                    list_vec.append(vec_ego_norm_bev.reshape(1, 20, 2))
            if len(list_vec):
                vecs = torch.cat(list_vec, dim=0).to(device)
                # print("vecs: ", vecs.shape)
                if self.dataset == 'av2' and i == 2:
                    HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=2.0)
                elif self.dataset == 'ruqi' and i == 4:
                    HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=2.0)
                else:
                    HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=2.0)
                rasterized_line = HARD_CUDA_RASTERIZER(vecs, int(self.bev_w), int(self.bev_h), 1.0)
                rasterized_line, _ = torch.max(rasterized_line, 0)
                rasterized_results[i] = rasterized_line
        # if len(self.global_map_dict[city_name][1]):
        #     sys.exit()
        return rasterized_results.permute(1, 2, 0).reshape(-1, self.num_classes)

    def get_global_map(self):
        return self.global_map_dict

    def save_global_map(self):
        if self.save_map_path is not None:
            global_map_dict = {}
            for city_name in self.city_list:
                global_map_dict[city_name] = {}
                for i in range(self.num_classes):
                    global_map_dict[city_name][i] = []
                    for vec in self.global_map_dict[city_name][i]:
                        global_map_dict[city_name][i].append(vec.tolist())
            with open(self.save_map_path, 'w') as f:
                json.dump(global_map_dict, f)
            f.close()