import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
from mmcv.utils import TORCH_VERSION, digit_version
from projects.mmdet3d_plugin.hrmap.global_map import GlobalMap
from projects.mmdet3d_plugin.hrmap.global_map_vec import GlobalMapVec
from projects.mmdet3d_plugin.hrmap.global_map_vec_tile import GlobalMapVecTile
import torch.distributed as dist
from projects.mmdet3d_plugin.maptr.modules.ops.diff_ras.polygon import SoftPolygon
from projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head import normalize_2d_pts
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from mmcv.runner import get_dist_info
import mmcv
import sys
import random


@DETECTORS.register_module()
class MapTRv2(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 global_map_cfg=None,
                 map_half_update=False,
                 map_quarter_update=False,
                 map_share=False,
                 map_state_prob=None,
                 map_mask_ratio=0.0,
                 map_update_num=None,
                 long_tail_dup=0,
                 list_long_tail_dup=[],
                 map_multi=0,
                 ):

        super(MapTRv2,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

        self.map_multi = map_multi
        self.i_gm = [i for i in range(self.map_multi)]
        if global_map_cfg is not None:
            self.global_map_type = global_map_cfg.get("global_map_type", "raster")
            if self.global_map_type == "raster":
                self.global_map = GlobalMap(global_map_cfg)
            elif self.global_map_type == "vector":
                self.global_map = GlobalMapVec(global_map_cfg)
            elif self.global_map_type == "vector_tile":
                if self.map_multi > 0:
                    # self.global_map = [GlobalMapVecTile(global_map_cfg)] * 4
                    global_map = GlobalMapVecTile(global_map_cfg)
                    self.global_map = [copy.deepcopy(global_map) for _ in range(self.map_multi)]
                else:
                    self.global_map = GlobalMapVecTile(global_map_cfg)
            else:
                raise NotImplementedError("not implemented global map type")
            self.update_map = global_map_cfg['update_map']
        else:
            self.global_map = None
            self.update_map = False
        self.epoch = -1
        self.iter = -1
        self.map_half_update = map_half_update
        self.map_quarter_update = map_quarter_update
        self.map_share = map_share
        self.map_state_prob = map_state_prob
        self.map_mask_ratio = map_mask_ratio
        self.map_update_num = map_update_num
        self.long_tail_dup = long_tail_dup
        self.list_long_tail_dup = list_long_tail_dup
        try:
            self.num_gpu = dist.get_world_size()
        except:
            self.num_gpu = 1

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_iter(self, iteration):
        self.iter = iteration

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          local_map=None,
                          gt_depth=None,
                          gt_seg_mask=None,
                          gt_pv_seg_mask=None,):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, local_map,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        depth = outs.pop('depth')
        losses = dict()
        # calculate depth loss
        if gt_depth is not None and depth is not None:
            loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(gt_depth, depth)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_depth = torch.nan_to_num(loss_depth)
            losses.update(loss_depth=loss_depth)

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        num_group = self.pts_bbox_head.num_group
        if num_group:
            lambda_group = 2.0 / (num_group + 1)
            for key, value in losses_pts.items():
                if key.endswith("cls") or key.endswith("pts") or key.endswith("dir"):
                    losses_pts[key] *= lambda_group

        losses.update(losses_pts)
        # import ipdb;ipdb.set_trace()
        k_one2many = self.pts_bbox_head.k_one2many
        if k_one2many > 0:
            multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
            multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
            for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
                each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
                each_gt_bboxes_3d.instance_labels = each_gt_bboxes_3d.instance_labels * k_one2many
                multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)
            # import ipdb;ipdb.set_trace()
            one2many_outs = outs['one2many_outs']
            loss_one2many_inputs = [multi_gt_bboxes_3d, multi_gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, one2many_outs]
            loss_dict_one2many = self.pts_bbox_head.loss(*loss_one2many_inputs, img_metas=img_metas)

            lambda_one2many = self.pts_bbox_head.lambda_one2many
            for key, value in loss_dict_one2many.items():
                if key + "_one2many" in losses.keys():
                    losses[key + "_one2many"] += value * lambda_one2many
                else:
                    losses[key + "_one2many"] = value * lambda_one2many
        
        if num_group:
            new_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
            new_gt_labels_3d = copy.deepcopy(gt_labels_3d)

            if self.long_tail_dup > 0:
                this_device = new_gt_labels_3d[0].device
                instance_added = []
                for each_gt_bboxes_3d in new_gt_bboxes_3d:
                    num_gt = len(each_gt_bboxes_3d.instance_list)
                    each_added = []
                    for i in range(num_gt):
                        if each_gt_bboxes_3d.instance_labels[i] in self.list_long_tail_dup:
                            for _ in range(self.long_tail_dup):
                                each_added.append(copy.deepcopy(each_gt_bboxes_3d.instance_list[i]))
                    instance_added.append(each_added)
                for i, each_added in enumerate(instance_added):
                    if len(each_added):
                        for ea in each_added:
                            new_gt_bboxes_3d[i].instance_list.append(ea)
                            new_gt_bboxes_3d[i].instance_labels.append(1)
                        label_added = torch.ones(len(each_added)).to(this_device).long()
                        new_gt_labels_3d[i] = torch.cat([new_gt_labels_3d[i], label_added], dim=0)

            loss_dict_group = self.pts_bbox_head.loss_group(new_gt_bboxes_3d, new_gt_labels_3d, 
                                                            list_preds_dicts=list(outs['group_outs'].values()))
            for key, value in loss_dict_group.items():
                losses[key] = value * lambda_group

        if self.update_map:
            new_map = self.pts_bbox_head.get_pred_mask(outs, self.global_map_type, status='train')
            # print("new_map: ", new_map.shape)
            self.update_global_map(img_metas, new_map, 'train')

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, return_map=False, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_map:
            return self.return_map()
        elif return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','local_map'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      gt_depth=None,
                      gt_seg_mask=None,
                      gt_pv_seg_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)

        len_queue = img.size(1)
        img = img[:, -1, ...]

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()

        if self.global_map is not None:
            # print("self.iter: ", self.iter)
            if self.map_multi > 0:
                for global_map in self.global_map:
                    global_map.check_map(img_feats[0].device, self.epoch, 'train', self.iter)
            else:
                self.global_map.check_map(img_feats[0].device, self.epoch, 'train', self.iter)
            local_map = self.obtain_global_map(img_metas, 'train', img_feats[0].device)
        else:
            local_map = None

        losses_pts = self.forward_pts_train(img_feats, lidar_feat, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, local_map, gt_depth,
                                            gt_seg_mask, gt_pv_seg_mask)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None,points=None,  **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], None, **kwargs)

        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    
    def simple_test_pts(self, x, lidar_feat, img_metas, local_map=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, local_map=local_map)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]

        if self.global_map is not None and self.update_map:
            new_map = self.pts_bbox_head.get_pred_mask(outs, self.global_map_type, status='val')
            # print("outs: ", outs['bev_embed'].device)
            # sys.exit()
            self.update_global_map(img_metas, new_map, 'val', outs['bev_embed'].device)
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        if self.global_map is not None:
            if self.map_multi > 0:
                for global_map in self.global_map:
                    global_map.check_map(img_feats[0].device, self.epoch, 'val', self.iter)
            else:
                self.global_map.check_map(img_feats[0].device, self.epoch, 'val', self.iter)
            local_map = self.obtain_global_map(img_metas, 'val', img_feats[0].device)
        else:
            local_map = None
        # if img_metas[0]['timestamp'] == '315966062559846000':
        # if local_map[..., 3].sum().item() > 0:
        #     print("local_map: ", local_map[..., 3].sum())
        #     print("img_metas: ", img_metas[0]['timestamp'])
        #     import matplotlib.pyplot as plt
        #     plt.imsave('tmp.jpg', local_map[..., 3].reshape(100, 200).cpu().numpy(), cmap='viridis')
        #     sys.exit()
        # import matplotlib.pyplot as plt
        # if local_map[..., 2].sum().item() > 0:
        #     plt.imsave('tmp.jpg', local_map[..., 2].reshape(self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w).cpu().numpy(), cmap='viridis')


        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, local_map, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

    def update_global_map(self, img_metas, raster, status, device=None):
        if self.global_map_type == 'raster':
            bs = raster.shape[0]
        elif self.global_map_type.startswith('vector'):
            bs = len(raster)
        else:
            raise NotImplementedError("not supported global map type")
        if self.map_half_update and self.training:
            bs = bs // 2
        if self.map_quarter_update and self.training:
            bs = bs // 4
        if self.map_update_num is not None and self.training:
            if self.map_update_num == 1.5:
                if torch.rand(1).item() < 0.5:
                    bs = 2
                else:
                    bs = 1
            if self.map_update_num < 1:
                if torch.rand(1).item() < self.map_update_num:
                    bs = 1
                else:
                    bs = 0
        for i in range(bs):
            metas = img_metas[i]
            city_name = metas['map_location']
            trans = metas['lidar2global']
            if self.map_share and not self.training and self.num_gpu > 1:
                if self.map_multi > 0:
                    city_name_index = self.global_map[0].city_list.index(city_name)
                else:
                    city_name_index = self.global_map.city_list.index(city_name)
                city_name_index = torch.tensor([city_name_index], dtype=torch.int32).to(device)
                city_name_index_gather = [torch.zeros_like(city_name_index) for _ in range(self.num_gpu)]
                dist.all_gather(city_name_index_gather, city_name_index)
                trans = torch.from_numpy(trans).to(device)
                trans_gather = [torch.zeros_like(trans) for _ in range(self.num_gpu)]
                dist.all_gather(trans_gather, trans)
                if self.global_map_type == "raster":
                    raster_gather = [torch.zeros_like(raster[i]) for _ in range(self.num_gpu)]
                    dist.all_gather(raster_gather, raster[i])
                else:
                    num_class = len(raster[i].keys())
                    vector_num = torch.zeros(num_class).to(device)
                    for k, v in raster[i].items():
                        vector_num[int(k)] = v.shape[0]
                    vector_num_gather = [torch.zeros_like(vector_num) for _ in range(self.num_gpu)]
                    dist.all_gather(vector_num_gather, vector_num)
                    vector_all = torch.zeros(self.pts_bbox_head.num_vec_one2one, \
                        self.pts_bbox_head.num_pts_per_vec, self.pts_bbox_head.code_size).to(device)
                    for k, v in raster[i].items():
                        idx_start, idx_end = int(vector_num[:int(k)].sum()), int(vector_num[:int(k)+1].sum())
                        if idx_end > idx_start:
                            vector_all[idx_start:idx_end] = v
                    vector_all_gather = [torch.zeros_like(vector_all) for _ in range(self.num_gpu)]
                    dist.all_gather(vector_all_gather, vector_all)              
                for j in range(self.num_gpu):
                    if self.map_multi > 0:
                        city_name_j = self.global_map[0].city_list[city_name_index_gather[j].item()]
                    else:
                        city_name_j = self.global_map.city_list[city_name_index_gather[j].item()]
                    trans_j = trans_gather[j].cpu().numpy()
                    if self.global_map_type == "raster":
                        self.global_map.update_map(city_name_j, trans_j, raster_gather[j], status)
                    else:
                        vector_num_j = vector_num_gather[j]
                        vector_all_j = vector_all_gather[j]
                        raster_j = {}
                        for idx_class in range(num_class):
                            idx_start, idx_end = int(vector_num_j[:idx_class].sum()), int(vector_num_j[:idx_class+1].sum())
                            raster_j[idx_class] = vector_all_j[idx_start:idx_end]
                        if self.map_multi > 0:
                            self.global_map[0].update_map(city_name_j, trans_j, raster_j, status)
                        else:
                            self.global_map.update_map(city_name_j, trans_j, raster_j, status)
            else:
                if self.map_multi > 0:
                    if not self.training:
                        self.global_map[0].update_map(city_name, trans, raster[i], status)
                    else:
                        # i_gm = random.randint(0, self.map_multi - 1)
                        # i_gm = (i + 1) * random.randint(1, self.map_multi // 4) - 1
                        # i_gm = self.i_gm[i % self.map_multi]
                        # i_gm = i    # mm4o4u
                        i_gm = i // 2    # mm4o2u
                        # i_gm = 0    # mm4o1u
                        self.global_map[i_gm].update_map(city_name, trans, raster[i], status)
                else:
                    self.global_map.update_map(city_name, trans, raster[i], status)

    def obtain_global_map(self, img_metas, status, device):
        bs = len(img_metas)
        bev_maps = []
        random.shuffle(self.i_gm)
        for i in range(bs):
            metas = img_metas[i]
            # print("metas: ", metas)
            # sys.exit()
            city_name = metas['map_location']
            trans = metas['lidar2global']
            timestamp = str(metas['timestamp'])
            if self.map_multi > 0:
                if not self.training:
                    local_map = self.global_map[0].get_map(city_name, trans, status, device, timestamp)
                else:
                    # if self.map_multi < 4:
                    #     i_gm = i // self.map_multi
                    # else:
                    #     i_gm = (i + 1) * random.randint(1, self.map_multi // 4) - 1
                    
                    # self.i_gm = random.randint(0, self.map_multi - 1)
                    # i_gm = self.i_gm[i % self.map_multi]
                    i_gm = i
                    local_map = self.global_map[i_gm].get_map(city_name, trans, status, device, timestamp)
            else:
                local_map = self.global_map.get_map(city_name, trans, status, device, timestamp)
            # print("local_map: ", local_map.shape)    # torch.Size([20000, 3])
            if self.map_state_prob is not None and self.training:
                assert len(self.map_state_prob) == 2
                this_prob = torch.rand(1).item()
                # empty
                if this_prob < self.map_state_prob[0]:
                    local_map_final = torch.zeros_like(local_map)
                # masked
                elif this_prob < self.map_state_prob[1]:
                    bev_h, bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
                    # print("bev_h, bev_w: ", bev_h, bev_w)
                    local_map_mask = torch.ones_like(local_map).view(bev_h, bev_w, 3)
                    local_map_mask[(bev_h - 20):, ...] = 0
                    local_map_final = local_map_mask.view(-1, 3) * local_map
                # complete
                else:
                    local_map_final = local_map
                bev_maps.append(local_map_final)
            elif self.map_mask_ratio and self.training:
                random_matrix = torch.rand_like(local_map)
                binary_matrix = (random_matrix > self.map_mask_ratio).float()
                local_map_final = binary_matrix * local_map
                bev_maps.append(local_map_final)
            else:
                bev_maps.append(local_map)
        bev_maps = torch.stack(bev_maps)
        bev_maps = bev_maps.permute(1, 0, 2)
        return bev_maps

    def return_map(self):
        if self.update_map:
            if self.map_multi > 0:
                self.global_map[0].save_global_map()
            else:
                self.global_map.save_global_map()
