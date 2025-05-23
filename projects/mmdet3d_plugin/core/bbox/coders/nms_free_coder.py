import torch
import sys

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from projects.mmdet3d_plugin.maptr.modules.ops.diff_ras.polygon import SoftPolygon


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
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    new_pts[...,2:3] = pts[...,2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h,patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
       
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


@BBOX_CODERS.register_module()
class MapTRNMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 z_cfg = dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 raster_threshold=0.4,    # 0.4
                 raster_threshold_val=0.7,
                 bev_h=200,
                 bev_w=100,
                 num_classes=10,
                 raster_type='default'):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.raster_threshold = raster_threshold
        self.raster_threshold_val = raster_threshold_val
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.z_cfg = z_cfg
        self.raster_type = raster_type

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]
       
        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range) 
        #num_q,num_p,2
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                        else denormalize_3d_pts(pts_preds, self.pc_range) 
        # final_box_preds = bbox_preds 
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :4] >=
                    self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <=
                     self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'pts': pts,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i],all_pts_preds[i]))
        return predictions_list

    def rasterize_preds(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]
        """
        # new_pts = pts.clone()
        # print("labels: ", labels)
        new_pts = pts[..., :2].clone()
        new_pts[..., 0:1] = pts[..., 0:1] * width
        # new_pts[..., 1:2] = (1.0 - pts[..., 1:2]) * height
        new_pts[..., 1:2] = (pts[..., 1:2]) * height

        divider_index = torch.nonzero(labels == 0, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 1, as_tuple=True)
        boundary_index = torch.nonzero(labels == 2, as_tuple=True)
        divider_pts = new_pts[divider_index]
        ped_crossing_pts = new_pts[ped_crossing_index]
        boundary_pts = new_pts[boundary_index]
        rasterized_results = torch.zeros(3, height, width, device=pts.device)
        if divider_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(divider_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[0] = rasterized_line

        if ped_crossing_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=inv_smoothness)
            rasterized_poly = HARD_CUDA_RASTERIZER(ped_crossing_pts, int(width), int(height), 1.0)
            rasterized_poly, _ = torch.max(rasterized_poly, 0)
            rasterized_results[1] = rasterized_poly

        if boundary_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(boundary_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[2] = rasterized_line

        if use_dilate:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            rasterized_results = max_pool(rasterized_results)

        return rasterized_results
    
    def rasterize_preds_v2(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]
        """
        # new_pts = pts.clone()
        # print("labels: ", labels)
        new_pts = pts[..., :2].clone()
        new_pts[..., 0:1] = pts[..., 0:1] * width
        # new_pts[..., 1:2] = (1.0 - pts[..., 1:2]) * height
        new_pts[..., 1:2] = (pts[..., 1:2]) * height

        divider_solid_index = torch.nonzero(labels == 0, as_tuple=True)
        divider_dashed_index = torch.nonzero(labels == 1, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 2, as_tuple=True)
        boundary_index = torch.nonzero(labels == 3, as_tuple=True)
        divider_solid_pts = new_pts[divider_solid_index]
        divider_dashed_pts = new_pts[divider_dashed_index]
        ped_crossing_pts = new_pts[ped_crossing_index]
        boundary_pts = new_pts[boundary_index]
        rasterized_results = torch.zeros(4, height, width, device=pts.device)
        if divider_solid_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(divider_solid_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[0] = rasterized_line
        
        if divider_dashed_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(divider_dashed_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[1] = rasterized_line

        if ped_crossing_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=inv_smoothness)
            rasterized_poly = HARD_CUDA_RASTERIZER(ped_crossing_pts, int(width), int(height), 1.0)
            rasterized_poly, _ = torch.max(rasterized_poly, 0)
            rasterized_results[2] = rasterized_poly

        if boundary_pts.shape[0] > 0:
            HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
            rasterized_line = HARD_CUDA_RASTERIZER(boundary_pts, int(width), int(height), 1.0)
            rasterized_line, _ = torch.max(rasterized_line, 0)
            rasterized_results[3] = rasterized_line

        if use_dilate:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            rasterized_results = max_pool(rasterized_results)

        return rasterized_results
    
    def rasterize_preds_v3(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]
        """
        # new_pts = pts.clone()
        # print("labels: ", labels)
        # new_pts = pts[..., :2].clone()
        # new_pts[..., 0:1] = pts[..., 0:1] * width
        # new_pts[..., 1:2] = (1.0 - pts[..., 1:2]) * height
        # new_pts[..., 1:2] = (pts[..., 1:2]) * height
        new_pts = denormalize_2d_pts(pts, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else denormalize_3d_pts(pts, self.pc_range)

        divider_solid_index = torch.nonzero(labels == 0, as_tuple=True)
        divider_dashed_index = torch.nonzero(labels == 1, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 2, as_tuple=True)
        boundary_index = torch.nonzero(labels == 3, as_tuple=True)
        # divider_solid_pts = new_pts[divider_solid_index]
        # divider_dashed_pts = new_pts[divider_dashed_index]
        # ped_crossing_pts = new_pts[ped_crossing_index]
        # boundary_pts = new_pts[boundary_index]
        # rasterized_results = torch.zeros(4, height, width, device=pts.device)
        # if divider_solid_pts.shape[0] > 0:
        #     HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
        #     rasterized_line = HARD_CUDA_RASTERIZER(divider_solid_pts, int(width), int(height), 1.0)
        #     rasterized_line, _ = torch.max(rasterized_line, 0)
        #     rasterized_results[0] = rasterized_line
        
        # if divider_dashed_pts.shape[0] > 0:
        #     HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
        #     rasterized_line = HARD_CUDA_RASTERIZER(divider_dashed_pts, int(width), int(height), 1.0)
        #     rasterized_line, _ = torch.max(rasterized_line, 0)
        #     rasterized_results[1] = rasterized_line

        # if ped_crossing_pts.shape[0] > 0:
        #     HARD_CUDA_RASTERIZER = SoftPolygon(mode="mask", inv_smoothness=inv_smoothness)
        #     rasterized_poly = HARD_CUDA_RASTERIZER(ped_crossing_pts, int(width), int(height), 1.0)
        #     rasterized_poly, _ = torch.max(rasterized_poly, 0)
        #     rasterized_results[2] = rasterized_poly

        # if boundary_pts.shape[0] > 0:
        #     HARD_CUDA_RASTERIZER = SoftPolygon(mode="boundary", inv_smoothness=inv_smoothness)
        #     rasterized_line = HARD_CUDA_RASTERIZER(boundary_pts, int(width), int(height), 1.0)
        #     rasterized_line, _ = torch.max(rasterized_line, 0)
        #     rasterized_results[3] = rasterized_line

        # if use_dilate:
        #     max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #     rasterized_results = max_pool(rasterized_results)

        rasterized_results = {}
        rasterized_results[0] = new_pts[divider_solid_index]
        rasterized_results[1] = new_pts[divider_dashed_index]
        rasterized_results[2] = new_pts[ped_crossing_index]
        rasterized_results[3] = new_pts[boundary_index]
        return rasterized_results
    
    def rasterize_preds_v4(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]
        """
        # map_classes = ['divider_solid_white', 'divider_dashed_white', 
        #        'divider_solid_yellow', 'divider_dashed_yellow', 
        #        'ped_crossing', 'boundary_fence', 'boundary_curb', 
        #        'centerline', 'stopline', 'road_marking', 'road_marking_straight', 
        #        'road_marking_left', 'road_marking_right', 'road_marking_uturn', 
        #        'road_marking_straight_left', 'road_marking_straight_right']
        
        new_pts = denormalize_2d_pts(pts, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else denormalize_3d_pts(pts, self.pc_range)

        divider_solid_white_index = torch.nonzero(labels == 0, as_tuple=True)
        divider_dashed_white_index = torch.nonzero(labels == 1, as_tuple=True)
        divider_solid_yellow_index = torch.nonzero(labels == 2, as_tuple=True)
        divider_dashed_yellow_index = torch.nonzero(labels == 3, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 4, as_tuple=True)
        boundary_fence_index = torch.nonzero(labels == 5, as_tuple=True)
        boundary_curb_index = torch.nonzero(labels == 6, as_tuple=True)
        centerline_index = torch.nonzero(labels == 7, as_tuple=True)
        stopline_index = torch.nonzero(labels == 8, as_tuple=True)
        road_marking_index = torch.nonzero(labels == 9, as_tuple=True)
        road_marking_straight_index = torch.nonzero(labels == 10, as_tuple=True)
        road_marking_left_index = torch.nonzero(labels == 11, as_tuple=True)
        road_marking_right_index = torch.nonzero(labels == 12, as_tuple=True)
        road_marking_uturn_index = torch.nonzero(labels == 13, as_tuple=True)
        road_marking_straight_left_index = torch.nonzero(labels == 14, as_tuple=True)
        road_marking_straight_right_index = torch.nonzero(labels == 15, as_tuple=True)

        rasterized_results = {}
        rasterized_results[0] = new_pts[divider_solid_white_index]
        rasterized_results[1] = new_pts[divider_dashed_white_index]
        rasterized_results[2] = new_pts[divider_solid_yellow_index]
        rasterized_results[3] = new_pts[divider_dashed_yellow_index]
        rasterized_results[4] = new_pts[ped_crossing_index]
        rasterized_results[5] = new_pts[boundary_fence_index]
        rasterized_results[6] = new_pts[boundary_curb_index]
        rasterized_results[7] = new_pts[centerline_index]
        rasterized_results[8] = new_pts[stopline_index]
        rasterized_results[9] = new_pts[road_marking_index]
        rasterized_results[10] = new_pts[road_marking_straight_index]
        rasterized_results[11] = new_pts[road_marking_left_index]
        rasterized_results[12] = new_pts[road_marking_right_index]
        rasterized_results[13] = new_pts[road_marking_uturn_index]
        rasterized_results[14] = new_pts[road_marking_straight_left_index]
        rasterized_results[15] = new_pts[road_marking_straight_right_index]

        return rasterized_results
    
    def rasterize_preds_v5(self, pts, labels, height=200, width=100, inv_smoothness=2.0, use_dilate=False):
        """
        Args:
            pts: shape [batch_size * num_pred, num_pts, 2]
        """
        new_pts = denormalize_2d_pts(pts, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else denormalize_3d_pts(pts, self.pc_range)

        divider_index = torch.nonzero(labels == 0, as_tuple=True)
        ped_crossing_index = torch.nonzero(labels == 1, as_tuple=True)
        boundary_index = torch.nonzero(labels == 2, as_tuple=True)

        rasterized_results = {}
        rasterized_results[0] = new_pts[divider_index]
        rasterized_results[1] = new_pts[ped_crossing_index]
        rasterized_results[2] = new_pts[boundary_index]
        return rasterized_results

    def decode_raster_single(self, cls_scores, bbox_preds, pts_preds, status='train'):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        # print("bbox_preds: ", bbox_preds.shape)
        # print("pts_preds: ", pts_preds.shape)
        # sys.exit()

        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
        #                 else denormalize_3d_pts(pts_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :4] >=
                    self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <=
                     self.post_center_range[4:]).all(1)

            # use score threshold
            if status == 'train':
                raster_threshold = self.raster_threshold
            elif status == 'val':
                raster_threshold = self.raster_threshold_val
            else:
                raise NotImplementedError('not supported status')
            if raster_threshold is not None:
                thresh_mask = final_scores > raster_threshold
                mask &= thresh_mask

            labels = final_preds[mask]
            if self.raster_type == 'default':
                raster_preds = self.rasterize_preds(pts_preds[mask], labels, self.bev_h, self.bev_w)
            elif self.raster_type == 'av2c4':
                raster_preds = self.rasterize_preds_v2(pts_preds[mask], labels, self.bev_h, self.bev_w)
            elif self.raster_type == 'av2c4vec':
                raster_preds = self.rasterize_preds_v3(pts_preds[mask], labels, self.bev_h, self.bev_w)
            elif self.raster_type == 'ruqic16vec':
                raster_preds = self.rasterize_preds_v4(pts_preds[mask], labels, self.bev_h, self.bev_w)
            elif self.raster_type == 'defaultvec':
                raster_preds = self.rasterize_preds_v5(pts_preds[mask], labels, self.bev_h, self.bev_w)
            else:
                raise NotImplementedError('not supported raster type for now!')
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return raster_preds

    def decode_raster(self, preds_dicts, status='train'):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1].detach()  # torch.Size([1, 100, 3])
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1].detach()  # torch.Size([1, 100, 4])
        all_pts_preds = preds_dicts['all_pts_preds'][-1].detach() # torch.Size([1, 100, 20, 2])
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_raster_single(all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i], status))
        return predictions_list