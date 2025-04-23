import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from mmcv.runner import get_dist_info
import torch.nn.functional as F
import sys
from shapely.geometry import box, LineString, Polygon
from projects.mmdet3d_plugin.maptr.modules.ops.diff_ras.polygon import SoftPolygon
import json
import math
import copy
from torch import nn
import os


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


class GlobalMapVecTile(nn.Module):
    def __init__(self, map_cfg):
        super().__init__()
        self.rasterize_scale = map_cfg.get('rasterize_scale', 1)
        self.bev_h = map_cfg['bev_h'] // self.rasterize_scale
        self.bev_w = map_cfg['bev_w'] // self.rasterize_scale
        self.pc_range = map_cfg['pc_range']
        self.load_map_path = map_cfg['load_map_path']
        self.save_map_path = map_cfg['save_map_path']
        self.num_classes = map_cfg['num_classes']
        self.bev_patch_h = self.pc_range[4] - self.pc_range[1]
        self.bev_patch_w = self.pc_range[3] - self.pc_range[0]
        self.dataset = map_cfg['dataset']
        if self.dataset == 'nuscenes':
            self.city_list = ['singapore-onenorth', 'boston-seaport',
                              'singapore-queenstown', 'singapore-hollandvillage']
            self.tile_offset = 1000
        elif self.dataset.startswith('av2'):
            self.city_list = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
            self.tile_offset = 10000
        elif self.dataset == 'ruqi':
            self.city_list = ['guangzhou']
            self.tile_offset = 0
        else:
            raise NotImplementedError("not supported dataset")
        self.tile_size = map_cfg.get('tile_size', 60)
        self.tile_interval = map_cfg.get('tile_interval', 0)
        self.tile_interval_get = map_cfg.get('tile_interval_get', 30)
        self.iter_based_reset = map_cfg.get('iter_based_reset', -1)
        inv_smoothness = map_cfg.get('inv_smoothness', None)
        self.inv_smoothness = {}
        for idx_class in range(self.num_classes):
            if inv_smoothness is None:
                self.inv_smoothness[idx_class] = 2.0
            else:
                self.inv_smoothness[idx_class] = inv_smoothness[idx_class]
        self.num_itpl = map_cfg.get('num_itpl', 20)
        self.global_map_dict = {}
        self.map_status = None
        self.epoch_point = -2
        self.iteration = 0
        self.update_interval = map_cfg.get('update_interval', 1)
        self.code_size = map_cfg.get('code_size', 3)

        self.use_time_prior = map_cfg.get('use_time_prior', True)
        self.use_map_prior = map_cfg.get('use_map_prior', False)
        if self.use_map_prior:
            self.load_gt_map_path = map_cfg['load_gt_map_path']
            self.load_gt_map()
        self.map_fuse_method = map_cfg.get('map_fuse_method', 'merge')    # merge or concat
        inv_smoothness_gt = map_cfg.get('inv_smoothness_gt', None)
        self.inv_smoothness_gt = {}
        for idx_class in range(self.num_classes):
            if inv_smoothness_gt is None:
                self.inv_smoothness_gt[idx_class] = 2.0
            else:
                self.inv_smoothness_gt[idx_class] = inv_smoothness_gt[idx_class]
        self.map_prior_displacement = map_cfg.get('map_prior_displacement', 0)
        self.map_prior_minor_revision = map_cfg.get('map_prior_minor_revision', 0)
        self.map_prior_major_revision = map_cfg.get('map_prior_major_revision', 0)
        self.map_prior_prob = map_cfg.get('map_prior_prob', 0.5)
        self.map_prior_noise = map_cfg.get('map_prior_noise', 5)
        self.use_map_prior_val = map_cfg.get('use_map_prior_val', True)
        self.use_time_prior_val = map_cfg.get('use_time_prior_val', True)
        
        self.save_map_prior = map_cfg.get('save_map_prior', None)
        self.save_time_prior = map_cfg.get('save_time_prior', None)

        # if self.dataset == 'av2':
        #     self.rasterize_mask = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[2])
        # elif self.dataset == 'ruqi':
        #     self.rasterize_mask = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[4])
        # elif self.dataset in ['nuscenes', 'av2c3']:
        #     self.rasterize_mask = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[1])
        # self.rasterize_line = SoftPolygon(mode="boundary", inv_smoothness=self.inv_smoothness[0])
    
    def load_map(self, device):
        with open(self.load_map_path, 'r') as f:
            global_map_dict = json.load(f)
        f.close()
        self.global_map_dict = {}
        for city_name in self.city_list:
            self.global_map_dict[city_name] = {}
            for idx_tile_x in global_map_dict[city_name].keys():
                self.global_map_dict[city_name][idx_tile_x] = {}
                for idx_tile_y in global_map_dict[city_name][idx_tile_x].keys():
                    self.global_map_dict[city_name][idx_tile_x][idx_tile_y] = {}
                    for idx_class in range(self.num_classes):
                        self.global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)] = []
                        for vec in global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)]:
                            self.global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)].append(torch.Tensor(vec))

    def load_gt_map(self):
        with open(self.load_gt_map_path, 'r') as f:
            global_map_dict = json.load(f)
        f.close()
        self.global_gt_map_dict = {}
        for city_name in self.city_list:
            self.global_gt_map_dict[city_name] = {}
            for timestamp in global_map_dict[city_name].keys():
                self.global_gt_map_dict[city_name][timestamp] = {}
                for idx_class in range(self.num_classes):
                    self.global_gt_map_dict[city_name][timestamp][str(idx_class)] = []
                    for vec in global_map_dict[city_name][timestamp][str(idx_class)]:
                        self.global_gt_map_dict[city_name][timestamp][str(idx_class)].append(torch.Tensor(vec))
            # for idx_tile_x in global_map_dict[city_name].keys():
            #     self.global_gt_map_dict[city_name][idx_tile_x] = {}
            #     for idx_tile_y in global_map_dict[city_name][idx_tile_x].keys():
            #         self.global_gt_map_dict[city_name][idx_tile_x][idx_tile_y] = {}
            #         for idx_class in range(self.num_classes):
            #             self.global_gt_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)] = []
            #             for vec in global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)]:
            #                 self.global_gt_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)].append(torch.Tensor(vec))


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
        elif self.iter_based_reset > 0:
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
        # 评测计数
        if self.map_status == 'val':
            self.iteration += 1

    def reset_map(self, iteration=-1):
        for city_name in self.city_list:
            print("reset map", city_name, "for epoch", self.epoch_point, \
                "status", self.map_status, "for iter", iteration)
            self.global_map_dict[city_name] = {}

    def create_map(self, device, status):
        for city_name in self.city_list:
            print("create map", city_name, status, "on", device, "for epoch", self.epoch_point)
            self.global_map_dict[city_name] = {}
        self.map_status = status

    def update_map(self, city_name, trans, raster, status):
        # print("self.iteration: ", self.iteration)
        if self.iteration % self.update_interval != 0:
            return
        trans = torch.from_numpy(trans).float()
        e2g_R, e2g_T = trans[:3, :3], trans[:3, -1].reshape(3, 1)
        # 计算瓦片索引
        idx_tile_x, interval_tile_x = \
            (e2g_T[0].item() + self.tile_offset) // self.tile_size, (e2g_T[0].item() + self.tile_offset) % self.tile_size
        idx_tile_y, interval_tile_y = \
            (e2g_T[1].item() + self.tile_offset) // self.tile_size, (e2g_T[1].item() + self.tile_offset) % self.tile_size
        list_idx_tile_x, list_idx_tile_y = [int(idx_tile_x)], [int(idx_tile_y)]
        if interval_tile_x < self.tile_interval:
            list_idx_tile_x.append(int(idx_tile_x - 1))
        elif interval_tile_x > (self.tile_size - self.tile_interval):
            list_idx_tile_x.append(int(idx_tile_x + 1))
        if interval_tile_y < self.tile_interval:
            list_idx_tile_y.append(int(idx_tile_y - 1))
        elif interval_tile_y > (self.tile_size - self.tile_interval):
            list_idx_tile_y.append(int(idx_tile_y + 1))
        # 双索引存储
        for idx_class in range(self.num_classes):
            for vec in raster[idx_class]:
                # if self.dataset == 'nuscenes':
                #     vec[:, 0], vec[:, 1] = vec[:, 1], vec[:, 0]
                if self.code_size == 2:
                    vec_ = torch.zeros(vec.shape[0], 3).to(vec.device)
                    vec_[:, :2] = vec
                    vec = vec_
                vec_global = (e2g_R @ vec.cpu().permute(1, 0) + e2g_T).permute(1, 0)
                for idx_tile_x in list_idx_tile_x:
                    if str(idx_tile_x) not in self.global_map_dict[city_name].keys():
                        self.global_map_dict[city_name][str(idx_tile_x)] = {}
                    for idx_tile_y in list_idx_tile_y:
                        if str(idx_tile_y) not in self.global_map_dict[city_name][str(idx_tile_x)].keys():
                            self.global_map_dict[city_name][str(idx_tile_x)][str(idx_tile_y)] = {}
                            for idx_class_init in range(self.num_classes):
                                self.global_map_dict[city_name][str(idx_tile_x)][str(idx_tile_y)][str(idx_class_init)] = []
                        self.global_map_dict[city_name][str(idx_tile_x)][str(idx_tile_y)][str(idx_class)].append(vec_global)

    def get_map(self, city_name, trans, status, device, timestamp=None):
        trans = torch.from_numpy(trans).float()
        e2g_R, e2g_T = trans[:3, :3], trans[:3, -1].reshape(3, 1)
        # 计算瓦片索引
        idx_tile_x, interval_tile_x = \
            (e2g_T[0].item() + self.tile_offset) // self.tile_size, (e2g_T[0].item() + self.tile_offset) % self.tile_size
        idx_tile_y, interval_tile_y = \
            (e2g_T[1].item() + self.tile_offset) // self.tile_size, (e2g_T[1].item() + self.tile_offset) % self.tile_size
        list_idx_tile_x, list_idx_tile_y = [int(idx_tile_x)], [int(idx_tile_y)]
        if interval_tile_x < self.tile_interval_get:
            list_idx_tile_x.append(int(idx_tile_x - 1))
        elif interval_tile_x > (self.tile_size - self.tile_interval_get):
            list_idx_tile_x.append(int(idx_tile_x + 1))
        if interval_tile_y < self.tile_interval_get:
            list_idx_tile_y.append(int(idx_tile_y - 1))
        elif interval_tile_y > (self.tile_size - self.tile_interval_get):
            list_idx_tile_y.append(int(idx_tile_y + 1))
        
        if self.use_time_prior:
            rasterized_results = torch.zeros(self.num_classes, self.bev_h, self.bev_w, device=device)
            # 训练阶段固定开启，推理阶段设置是否开启
            if (status == 'val' and self.use_time_prior_val) or (status == 'train'):
                # 初始化局部矢量分类存放字典
                dict_vec = {}
                for idx_class in range(self.num_classes):
                    dict_vec[idx_class] = []
                # 双索引提取
                for idx_tile_x in list_idx_tile_x:
                    if str(idx_tile_x) not in self.global_map_dict[city_name].keys():
                        # return rasterized_results.permute(1, 2, 0).reshape(-1, self.num_classes)
                        continue
                    for idx_tile_y in list_idx_tile_y:
                        if str(idx_tile_y) not in self.global_map_dict[city_name][str(idx_tile_x)].keys():
                            # return rasterized_results.permute(1, 2, 0).reshape(-1, self.num_classes)
                            continue
                        global_map_dict_tile = copy.deepcopy(self.global_map_dict[city_name][str(idx_tile_x)][str(idx_tile_y)])
                        g2e_R, g2e_T = e2g_R.T, - e2g_R.T @ e2g_T
                        xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
                        # patch_ego_xy = torch.Tensor([[xmin, ymin, 0], [xmin, ymax, 0], [xmax, ymax, 0], [xmax, ymin, 0], [xmin, ymin, 0]])
                        patch_ego_xy = torch.Tensor([[xmin, ymin, zmin], [xmin, ymax, zmin], \
                            [xmax, ymax, zmin], [xmax, ymin, zmin], [xmin, ymin, zmin], \
                            [xmin, ymin, zmax], [xmin, ymax, zmax], \
                            [xmax, ymax, zmax], [xmax, ymin, zmax], [xmin, ymin, zmax]])
                        # if self.dataset == 'nuscenes':
                        #     patch_ego_xy[:, 0], patch_ego_xy[:, 1] = patch_ego_xy[:, 1], patch_ego_xy[:, 0]
                        patch_global_xy = e2g_R @ patch_ego_xy.permute(1, 0) + e2g_T
                        # patch_global_xy = Polygon(patch_global_xy.permute(1, 0)[..., :2].tolist())
                        # for idx_class in range(self.num_classes):
                        #     for vec in global_map_dict_tile[str(idx_class)]:
                        #         vec_xy = LineString(vec[..., :2])
                        #         if vec_xy.intersects(patch_global_xy):
                        #             vec_ego = g2e_R @ vec.permute(1, 0) + g2e_T
                        #             vec_ego_norm = normalize_3d_pts(vec_ego.permute(1, 0), self.pc_range)
                        #             # 确认可以不截断，栅格化会自动忽略边界外的矢量
                        #             vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
                        #             vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w
                        #             vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h
                        #             dict_vec[idx_class].append(vec_ego_norm_bev.reshape(1, 20, 2))
                        # 加速提取 & 高程限制
                        # patch_ego_yz = torch.Tensor([[0, ymin, zmin], [0, ymin, zmax], [0, ymax, zmax], [0, ymax, zmin], [0, ymin, zmin]])
                        # patch_global_yz = e2g_R @ patch_ego_yz.permute(1, 0) + e2g_T
                        for idx_class in range(self.num_classes):
                            if len(global_map_dict_tile[str(idx_class)]):
                                # if len(global_map_dict_tile[str(idx_class)]) > 100 and status == 'train':
                                #     print(idx_class, len(global_map_dict_tile[str(idx_class)]))
                                # 加速推断
                                # if len(global_map_dict_tile[str(idx_class)]) > 50 and status == 'val':
                                #     global_map_dict_tile[str(idx_class)] = global_map_dict_tile[str(idx_class)][-50:]
                                vec_all = torch.cat(global_map_dict_tile[str(idx_class)]).reshape(-1, 20, 3)
                                if self.num_itpl != 20:
                                    vec_all_itpl = F.interpolate(vec_all.permute(0, 2, 1).reshape(-1, 1, 20), size=self.num_itpl, \
                                        mode='linear', align_corners=True)
                                    vec_all_itpl = vec_all_itpl.reshape(-1, 3, self.num_itpl).permute(0, 2, 1)
                                else:
                                    vec_all_itpl = vec_all.clone()
                                check_point_xy = (vec_all_itpl[..., 0] > patch_global_xy[0].min()) & \
                                    (vec_all_itpl[..., 0] < patch_global_xy[0].max()) & \
                                    (vec_all_itpl[..., 1] > patch_global_xy[1].min()) & \
                                    (vec_all_itpl[..., 1] < patch_global_xy[1].max())
                                check_instance_xy = torch.nonzero(check_point_xy.float().max(1)[0]).reshape(-1).tolist()
                                # check_instance = check_instance_xy
                                # check_point_yz = (vec_all[..., 1] > patch_global_yz[1].min()) & (vec_all[..., 1] < patch_global_yz[1].max()) \
                                #     & (vec_all[..., 2] > patch_global_yz[2].min()) & (vec_all[..., 2] < patch_global_yz[2].max())
                                # check_instance_yz = torch.nonzero(check_point_yz.float().max(1)[0]).reshape(-1).tolist()
                                # check_instance = list(set(check_instance_xy) & set(check_instance_yz))
                                # if len(check_instance):
                                if len(check_instance_xy):
                                    vec_xy_itpl = vec_all_itpl[check_instance_xy]
                                    hypotenuse = torch.norm(vec_xy_itpl - e2g_T.reshape(1, 1, 3), dim=2)
                                    leg = vec_xy_itpl[..., -1] - e2g_T[-1]
                                    tan = (leg / hypotenuse).abs()
                                    # 如果地面元素任意点与自车中心连线矢量与地面夹角大于20度，则过滤
                                    check_instance_z = torch.nonzero(torch.eq((tan > 0.364).sum(dim=1), 0)).reshape(-1).tolist()
                                    if len(check_instance_z):
                                        vec_ego = g2e_R @ vec_all[check_instance_xy][check_instance_z].reshape(-1, 3).permute(1, 0) + g2e_T
                                        vec_ego = vec_ego.permute(1, 0)
                                        # if self.dataset == 'nuscenes':
                                        #     vec_ego[:, 0], vec_ego[:, 1] = vec_ego[:, 1], vec_ego[:, 0]
                                        vec_ego_norm = normalize_3d_pts(vec_ego, self.pc_range)
                                        vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
                                        vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w
                                        vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h
                                        dict_vec[idx_class].append(vec_ego_norm_bev.reshape(-1, 20, 2))
                                # if len(check_instance_xy):
                                #     vec_ego = g2e_R @ vec_all[check_instance_xy].reshape(-1, 3).permute(1, 0) + g2e_T
                                #     vec_ego_norm = normalize_3d_pts(vec_ego.permute(1, 0), self.pc_range)
                                #     vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
                                #     vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w
                                #     vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h
                                #     dict_vec[idx_class].append(vec_ego_norm_bev.reshape(-1, 20, 2))
                
                if status == 'val' and self.save_time_prior is not None:
                    dict_time_prior = {}

                # 局部矢量分类栅格化
                for idx_class in range(self.num_classes):
                    if len(dict_vec[idx_class]):
                        vecs = torch.cat(dict_vec[idx_class], dim=0).to(device)
                        if status == 'val' and self.save_time_prior is not None:
                            dict_time_prior[str(idx_class)] = vecs.cpu().tolist()
                        if self.dataset == 'av2' and idx_class == 2:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[idx_class])
                        elif self.dataset == 'ruqi' and idx_class in [4, 9, 10, 11, 12, 13, 14, 15]:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[idx_class])
                        elif self.dataset in ['nuscenes', 'av2c3'] and idx_class == 1:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness[idx_class])
                        else:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=self.inv_smoothness[idx_class])
                        # if self.dataset == 'av2' and idx_class == 2:
                        #     HARD_CUDA_RASTERIZER = self.rasterize_mask
                        # elif self.dataset == 'ruqi' and idx_class in [4, 9, 10, 11, 12, 13, 14, 15]:
                        #     HARD_CUDA_RASTERIZER = self.rasterize_mask
                        # elif self.dataset in ['nuscenes', 'av2c3'] and idx_class == 1:
                        #     HARD_CUDA_RASTERIZER = self.rasterize_mask
                        # else:
                        #     HARD_CUDA_RASTERIZER = self.rasterize_line
                        rasterized_line = HARD_CUDA_RASTERIZER(vecs, int(self.bev_w), int(self.bev_h), 1.0)
                        
                        rasterized_line, _ = torch.max(rasterized_line, 0)
                        rasterized_results[idx_class] = rasterized_line
                
                if status == 'val' and self.save_time_prior is not None:
                    dir_time_prior = os.path.join(self.save_time_prior, str(timestamp) + '.json')
                    with open(dir_time_prior, 'w') as f:
                        json.dump(dict_time_prior, f)
                    f.close()
            
            if self.rasterize_scale > 1:
                rasterized_results = F.interpolate(rasterized_results.unsqueeze(0), \
                    scale_factor=self.rasterize_scale, mode='bicubic', align_corners=True)[0]
        
        # if self.use_map_prior:
        #     rasterized_gts = torch.zeros(self.num_classes, self.bev_h, self.bev_w, device=device)
        #     # 训练阶段概率开启，推理阶段设置是否开启
        #     if (status == 'val' and self.use_map_prior_val) or (status == 'train' and torch.rand(1) < self.map_prior_prob):
        #         # 初始化局部矢量分类存放字典
        #         dict_gt_vec = {}
        #         for idx_class in range(self.num_classes):
        #             dict_gt_vec[idx_class] = []
        #         # 双索引提取
        #         for idx_tile_x in list_idx_tile_x:
        #             if str(idx_tile_x) not in self.global_gt_map_dict[city_name].keys():
        #                 continue
        #             for idx_tile_y in list_idx_tile_y:
        #                 if str(idx_tile_y) not in self.global_gt_map_dict[city_name][str(idx_tile_x)].keys():
        #                     continue
        #                 global_map_dict_tile = copy.deepcopy(self.global_gt_map_dict[city_name][str(idx_tile_x)][str(idx_tile_y)])
        #                 g2e_R, g2e_T = e2g_R.T, - e2g_R.T @ e2g_T
        #                 xmin, ymin, zmin, xmax, ymax, zmax = self.pc_range
        #                 patch_ego_xy = torch.Tensor([[xmin, ymin, zmin], [xmin, ymax, zmin], \
        #                     [xmax, ymax, zmin], [xmax, ymin, zmin], [xmin, ymin, zmin], \
        #                     [xmin, ymin, zmax], [xmin, ymax, zmax], \
        #                     [xmax, ymax, zmax], [xmax, ymin, zmax], [xmin, ymin, zmax]])
        #                 patch_global_xy = e2g_R @ patch_ego_xy.permute(1, 0) + e2g_T
        #                 for idx_class in range(self.num_classes):
        #                     if len(global_map_dict_tile[str(idx_class)]):
        #                         # if len(global_map_dict_tile[str(idx_class)]) > 100 and status == 'train':
        #                         #     print(idx_class, len(global_map_dict_tile[str(idx_class)]))
        #                         # 加速推断
        #                         # if len(global_map_dict_tile[str(idx_class)]) > 50 and status == 'val':
        #                         #     global_map_dict_tile[str(idx_class)] = global_map_dict_tile[str(idx_class)][-50:]
        #                         vec_all = torch.cat(global_map_dict_tile[str(idx_class)]).reshape(-1, 20, 3)
        #                         if self.num_itpl > 20:
        #                             vec_all_itpl = F.interpolate(vec_all.permute(0, 2, 1).reshape(-1, 1, 20), size=self.num_itpl, \
        #                                 mode='linear', align_corners=False)
        #                             vec_all_itpl = vec_all_itpl.reshape(-1, 3, self.num_itpl).permute(0, 2, 1)
        #                         else:
        #                             vec_all_itpl = vec_all.clone()
        #                         check_point_xy = (vec_all_itpl[..., 0] > patch_global_xy[0].min()) & \
        #                             (vec_all_itpl[..., 0] < patch_global_xy[0].max()) & \
        #                             (vec_all_itpl[..., 1] > patch_global_xy[1].min()) & \
        #                             (vec_all_itpl[..., 1] < patch_global_xy[1].max())
        #                         check_instance_xy = torch.nonzero(check_point_xy.float().max(1)[0]).reshape(-1).tolist()
        #                         if len(check_instance_xy):
        #                             vec_xy_itpl = vec_all_itpl[check_instance_xy]
        #                             hypotenuse = torch.norm(vec_xy_itpl - e2g_T.reshape(1, 1, 3), dim=2)
        #                             leg = vec_xy_itpl[..., -1] - e2g_T[-1]
        #                             tan = (leg / hypotenuse).abs()
        #                             # 如果地面元素任意点与自车中心连线矢量与地面夹角大于20度，则过滤
        #                             check_instance_z = torch.nonzero(torch.eq((tan > 0.364).sum(dim=1), 0)).reshape(-1).tolist()
        #                             if len(check_instance_z):
        #                                 vec_ego = g2e_R @ vec_all[check_instance_xy][check_instance_z].reshape(-1, 3).permute(1, 0) + g2e_T
        #                                 vec_ego = vec_ego.permute(1, 0)
        #                                 vec_ego_norm = normalize_3d_pts(vec_ego, self.pc_range)
        #                                 vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
        #                                 vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w
        #                                 vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h
        #                                 dict_gt_vec[idx_class].append(vec_ego_norm_bev.reshape(-1, 20, 2))
        
        if self.use_map_prior:
            rasterized_gts = torch.zeros(self.num_classes, int(self.bev_h * self.rasterize_scale), \
                int(self.bev_w * self.rasterize_scale), device=device)
            # 训练阶段概率开启，推理阶段设置是否开启
            if (status == 'val' and self.use_map_prior_val) or \
                (status == 'train' and torch.rand(1) < self.map_prior_prob and rasterized_results.max() > 0.8):
                # 初始化局部矢量分类存放字典
                dict_gt_vec = {}
                for idx_class in range(self.num_classes):
                    dict_gt_vec[idx_class] = []
                # timestamp索引提取
                global_map_dict_tile = copy.deepcopy(self.global_gt_map_dict[city_name][str(timestamp)])

                for idx_class in range(self.num_classes):
                    num_vec = len(global_map_dict_tile[str(idx_class)])
                    if num_vec:
                        vec_all = torch.cat(global_map_dict_tile[str(idx_class)]).reshape(num_vec, 20, -1)
                        if self.code_size == 2 and self.dataset == 'nuscenes':
                            vec_ego = torch.zeros(num_vec, 20, 3).to(vec_all.device)
                            vec_ego[..., :2] = vec_all
                        else:
                            vec_ego = vec_all
                        vec_ego_norm = normalize_3d_pts(vec_ego, self.pc_range)
                        vec_ego_norm_bev = vec_ego_norm[..., :2].clone()
                        vec_ego_norm_bev[..., 0:1] = vec_ego_norm[..., 0:1] * self.bev_w * self.rasterize_scale
                        vec_ego_norm_bev[..., 1:2] = vec_ego_norm[..., 1:2] * self.bev_h * self.rasterize_scale
                        dict_gt_vec[idx_class].append(vec_ego_norm_bev.reshape(-1, 20, 2))
                
                if status == 'val' and self.map_prior_displacement > 0:
                    displacement_noise = torch.rand(1, 1, 2).to(device) * self.map_prior_displacement
                
                if status == 'val' and self.save_map_prior is not None:
                    dict_map_prior = {}

                # 局部矢量分类栅格化
                for idx_class in range(self.num_classes):
                    if len(dict_gt_vec[idx_class]):
                        vecs = torch.cat(dict_gt_vec[idx_class], dim=0).to(device)
                        if status == 'val' and self.map_prior_displacement > 0:
                            vecs = vecs + displacement_noise
                        if status == 'val' and self.map_prior_minor_revision > 0 and vecs.shape[0] > 0:
                            num_noise = torch.randint(1, vecs.shape[0] + 1, (1,)).item()
                            idx_noise = torch.randperm(vecs.shape[0])[:num_noise]
                            minor_revision_noise = torch.rand(num_noise, 1, 2).to(device) * self.map_prior_minor_revision
                            vecs[idx_noise] = vecs[idx_noise] + minor_revision_noise
                        if status == 'val' and self.map_prior_major_revision > 0:
                            major_revision_noise = torch.rand(vecs.shape[0], 1, 2).to(device) * self.map_prior_major_revision
                            vecs = vecs + major_revision_noise
                        if status == 'train' and self.map_prior_noise > 0 and vecs.shape[0] > 0:
                            num_noise = torch.randint(1, vecs.shape[0] + 1, (1,)).item()
                            idx_noise = torch.randperm(vecs.shape[0])[:num_noise]
                            noise_noise = torch.rand(num_noise, 1, 2).to(device) * self.map_prior_noise
                            vecs[idx_noise] = vecs[idx_noise] + noise_noise
                        
                        if status == 'val' and self.save_map_prior is not None:
                            dict_map_prior[str(idx_class)] = vecs.cpu().tolist()

                        if self.dataset == 'av2' and idx_class == 2:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness_gt[idx_class])
                        elif self.dataset == 'ruqi' and idx_class in [4, 9, 10, 11, 12, 13, 14, 15]:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness_gt[idx_class])
                        elif self.dataset in ['nuscenes', 'av2c3'] and idx_class == 1:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=self.inv_smoothness_gt[idx_class])
                        else:
                            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=self.inv_smoothness_gt[idx_class])
                        rasterized_line = HARD_CUDA_RASTERIZER(vecs, int(self.bev_w * self.rasterize_scale), \
                            int(self.bev_h * self.rasterize_scale), 1.0)
                        rasterized_line, _ = torch.max(rasterized_line, 0)
                        rasterized_gts[idx_class] = rasterized_line
        
                if status == 'val' and self.save_map_prior is not None:
                    dir_map_prior = os.path.join(self.save_map_prior, str(timestamp) + '.json')
                    with open(dir_map_prior, 'w') as f:
                        json.dump(dict_map_prior, f)
                    f.close()
        
        if self.use_time_prior and self.use_map_prior:
            if self.map_fuse_method == 'merge':
                rasterized_results = torch.max(rasterized_results, rasterized_gts)
            elif self.map_fuse_method == 'concat':
                rasterized_results = torch.cat([rasterized_results, rasterized_gts], dim=0)
                return rasterized_results.permute(1, 2, 0).reshape(-1, self.num_classes * 2)
            else:
                raise NotImplementedError('either merge or concat')
        
        if not self.use_time_prior and self.use_map_prior:
            rasterized_results = rasterized_gts

        return rasterized_results.permute(1, 2, 0).reshape(-1, self.num_classes)

    def get_global_map(self):
        return self.global_map_dict

    def save_global_map(self):
        if self.save_map_path is not None:
            global_map_dict = {}
            for city_name in self.city_list:
                global_map_dict[city_name] = {}
                for idx_tile_x in self.global_map_dict[city_name].keys():
                    global_map_dict[city_name][idx_tile_x] = {}
                    for idx_tile_y in self.global_map_dict[city_name][idx_tile_x].keys():
                        global_map_dict[city_name][idx_tile_x][idx_tile_y] = {}
                        for idx_class in range(self.num_classes):
                            global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)] = []
                            for vec in self.global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)]:
                                global_map_dict[city_name][idx_tile_x][idx_tile_y][str(idx_class)].append(vec.tolist())
            with open(self.save_map_path, 'w') as f:
                json.dump(global_map_dict, f)
            f.close()