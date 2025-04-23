import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn.bricks.transformer import MultiheadAttention
# from mmcv.ops.multi_scale_deform_attn import CustomMSDeformableAttention
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] -
                                          pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] -
                                          pc_range[1]) + pc_range[1])
    new_pts[..., 2:3] = (pts[..., 2:3] * (pc_range[5] -
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


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] -
                                              pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] -
                                              pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] -
                                          pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] -
                                          pc_range[1]) + pc_range[1])
    return new_pts


@HEADS.register_module()
class MapTRv2Head(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 #  num_vec=20,
                 num_vec_one2one=50,
                 num_vec_one2many=0,
                 k_one2many=0,
                 num_group=0,
                 lambda_one2many=1,
                 use_dino=False,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 map_query_scale=2,
                 query_init_attn='hrmapnet',
                 query_init_attn_np=4,
                 use_map_prior=False,
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 aux_seg=dict(
                     use_aux_seg=False,
                     bev_seg=False,
                     pv_seg=False,
                     seg_classes=1,
                     feat_down_sample=32,
                 ),
                 z_cfg=dict(
                     pred_z_flag=False,
                     gt_z_flag=False,
                 ),
                 loss_pts=dict(type='ChamferDistance',
                               loss_src_weight=1.0,
                               loss_dst_weight=1.0),
                 loss_seg=dict(type='SimpleLoss',
                               pos_weight=2.13,
                               loss_weight=1.0),
                 loss_pv_seg=dict(type='SimpleLoss',
                                  pos_weight=2.13,
                                  loss_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = 2 if not z_cfg['pred_z_flag'] else 3
        else:
            self.code_size = 2
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        if 'pts' not in self.query_embed_type:
            self.code_size = self.code_size * num_pts_per_vec
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.aux_seg = aux_seg
        self.z_cfg = z_cfg

        self.use_dino = use_dino
        self.query_init_attn = query_init_attn
        self.query_init_attn_np = query_init_attn_np
        self.use_map_prior = use_map_prior

        super(MapTRv2Head, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.num_group = num_group

        self.loss_seg = build_loss(loss_seg)
        self.loss_pv_seg = build_loss(loss_pv_seg)

        self.map_query_scale = map_query_scale
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        if 'pts' not in self.query_embed_type:
            reg_branch.append(Linear(self.embed_dims, self.embed_dims * 2))
            reg_branch.append(nn.ReLU())
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims * 2, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)
        else:
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if self.aux_seg['use_aux_seg']:
            if not (self.aux_seg['bev_seg'] or self.aux_seg['pv_seg']):
                raise ValueError('aux_seg must have bev_seg or pv_seg')
            if self.aux_seg['bev_seg']:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg['seg_classes'], kernel_size=1, padding=0)
                )
            if self.aux_seg['pv_seg']:
                self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg['seg_classes'], kernel_size=1, padding=0)
                )

        if not self.as_two_stage:
            if 'BEVFormerEncoder' in self.bev_encoder_type or self.transformer.map_encoder is not None:
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)
            elif self.query_embed_type == 'instance':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = None
            elif self.query_embed_type == 'map_pts':
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
                if self.use_map_prior:
                    self.label_encoder = nn.Linear(self.cls_out_channels * 2, self.embed_dims)
                else:
                    self.label_encoder = nn.Linear(self.cls_out_channels, self.embed_dims)
                if self.query_init_attn == 'hrmapnet':
                    self.query_attention = MultiheadAttention(self.embed_dims, num_heads=8, \
                        batch_first=True, dropout=0.0)
                elif self.query_init_attn.startswith('unippmap'):
                    self.query_attention = CustomMSDeformableAttention(self.embed_dims, num_heads=8, \
                        batch_first=True, dropout=0.0, num_levels=1, num_points=self.query_init_attn_np)    # 8    
                else:
                    raise NotImplementedError("not supported query init attn!!!")
            elif self.query_embed_type == 'map_instance':
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
                self.pts_embedding = None
                self.label_encoder = nn.Linear(self.cls_out_channels, self.embed_dims)
                self.query_attention = MultiheadAttention(self.embed_dims,
                                                          num_heads=8,
                                                          batch_first=True,
                                                          dropout=0.1)
            elif self.query_embed_type == 'map_instance_pos':
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = None
                self.label_encoder = nn.Linear(self.cls_out_channels, self.embed_dims)
                self.query_attention = MultiheadAttention(self.embed_dims * 2,
                                                          num_heads=8,
                                                          batch_first=True,
                                                          dropout=0.1)
        
        if self.use_dino:
            self.box_encoder = nn.Linear(self.code_size, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if 'map' in self.query_embed_type:
            xavier_init(self.label_encoder, distribution='uniform', bias=0.)
        if self.use_dino:
            xavier_init(self.box_encoder, distribution='uniform', bias=0.)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'local_map'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, local_map=None, only_bev=False,
                gt_bboxes_3d=None, gt_labels_3d=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
            # import ipdb;ipdb.set_trace()

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import ipdb;ipdb.set_trace()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        elif self.query_embed_type == 'instance':
            object_query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
        if self.bev_embedding is not None:
            # for map encoder
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if self.query_embed_type == 'map_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)
            query = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

            local_map_input = local_map.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
            local_map_input = F.interpolate(local_map_input,
                                            [self.bev_h // self.map_query_scale,
                                             self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
            local_map_pos = F.interpolate(bev_pos,
                                          [self.bev_h // self.map_query_scale,
                                           self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)

            if self.query_init_attn == 'hrmapnet':
                local_map_valid, _ = torch.max(local_map_input, dim=-1)
                ###!!! For debug only
                # local_map_valid = torch.rand(local_map_valid.shape)
                query_embeds_list = []
                for i in range(bs):
                    valid_index = local_map_valid[i] > 0.8
                    # valid_index为零和不为零看起来完全是两个模式，感觉可以尝试mask_ratio
                    if torch.equal(valid_index,
                                torch.zeros(valid_index.shape, dtype=torch.bool, device=valid_index.device)):
                        query_embeds_list.append(query.clone())
                    else:
                        map_pos_embed = local_map_pos[i][valid_index]
                        map_label = local_map_input[i][valid_index]
                        map_label_embed = self.label_encoder(map_label)
                        map_query = map_pos_embed + map_label_embed
                        query_embeds = self.query_attention(query.unsqueeze(0), key=map_query.unsqueeze(0))
                        query_embeds_list.append(query_embeds.squeeze(0))
                object_query_embeds = torch.stack(query_embeds_list, dim=0)
            elif self.query_init_attn == 'unippmap':
                num_token = self.bev_h // self.map_query_scale * self.bev_w // self.map_query_scale
                map_label_embed = self.label_encoder(local_map_input.view(bs, num_token, -1))
                map_pos_embed = local_map_pos.view(bs, num_token, -1)
                map_query = map_pos_embed + map_label_embed
                query_input = query.unsqueeze(0).repeat(bs, 1, 1)
                reference_point_input = self.transformer.reference_points(query_input)
                object_query_embeds = self.query_attention(query_input, key=None, value=map_query, 
                    reference_points = reference_point_input[..., :2].unsqueeze(-2), 
                    spatial_shapes=torch.tensor([[self.bev_h // self.map_query_scale, self.bev_w // self.map_query_scale]], device=map_query.device),
                    level_start_index=torch.tensor([0], device=map_query.device),)
            elif self.query_init_attn == 'unippmapv2':
                # if self.training:
                #     local_map_valid, _ = torch.max(local_map_input[..., :self.cls_out_channels], dim=-1)
                # else:
                #     local_map_valid, _ = torch.max(local_map_input, dim=-1)
                local_map_valid, _ = torch.max(local_map_input, dim=-1)
                num_token = self.bev_h // self.map_query_scale * self.bev_w // self.map_query_scale
                reference_point_input = self.transformer.reference_points(query.unsqueeze(0))
                spatial_shapes_input = torch.tensor([[self.bev_h // self.map_query_scale, self.bev_w // self.map_query_scale]], device=query.device)
                level_start_index_input = torch.tensor([0], device=query.device)
                query_embeds_list = []
                for i in range(bs):
                    valid_index = local_map_valid[i] > 0.8
                    if torch.equal(valid_index,
                                torch.zeros(valid_index.shape, dtype=torch.bool, device=valid_index.device)):
                        query_embeds_list.append(query.clone())
                    else:
                        map_label_embed = self.label_encoder(local_map_input[i].view(1, num_token, -1))
                        map_pos_embed = local_map_pos[i].view(1, num_token, -1)
                        map_query = map_pos_embed + map_label_embed
                        query_embeds = self.query_attention(query.unsqueeze(0), key=None, value=map_query, 
                            reference_points = reference_point_input[..., :2].unsqueeze(-2), 
                            spatial_shapes=spatial_shapes_input,
                            level_start_index=level_start_index_input,)
                        query_embeds_list.append(query_embeds.squeeze(0))
                object_query_embeds = torch.stack(query_embeds_list, dim=0)
            else:
                raise NotImplementedError("not supported query init attn!!!")
        
        elif self.query_embed_type == 'map_instance':
            local_map_input = local_map.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
            local_map_input = F.interpolate(local_map_input,
                                            [self.bev_h // self.map_query_scale,
                                             self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
            local_map_pos = F.interpolate(bev_pos,
                                          [self.bev_h // self.map_query_scale,
                                           self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
            local_map_valid, _ = torch.max(local_map_input, dim=-1)
            query_embeds_list = []
            for i in range(bs):
                valid_index = local_map_valid[i] > 0.8
                if torch.equal(valid_index,
                               torch.zeros(valid_index.shape, dtype=torch.bool, device=valid_index.device)):
                    query_embeds_list.append(self.instance_embedding.weight[0:num_vec].to(dtype))
                else:
                    map_pos_embed = local_map_pos[i][valid_index]
                    map_label = local_map_input[i][valid_index]
                    map_label_embed = self.label_encoder(map_label)
                    map_query = map_pos_embed + map_label_embed
                    query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
                    query_embeds = self.query_attention(query_embeds.unsqueeze(0), key=map_query.unsqueeze(0))
                    query_embeds_list.append(query_embeds.squeeze(0))
            object_query_embeds = torch.stack(query_embeds_list, dim=0)
        elif self.query_embed_type == 'map_instance_pos':
            local_map_input = local_map.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
            local_map_input = F.interpolate(local_map_input,
                                            [self.bev_h // self.map_query_scale,
                                             self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
            local_map_pos = F.interpolate(bev_pos,
                                          [self.bev_h // self.map_query_scale,
                                           self.bev_w // self.map_query_scale]).permute(0, 2, 3, 1)
            local_map_valid, _ = torch.max(local_map_input, dim=-1)
            query_embeds_list = []
            for i in range(bs):
                valid_index = local_map_valid[i] > 0.8
                if torch.equal(valid_index,
                               torch.zeros(valid_index.shape, dtype=torch.bool, device=valid_index.device)):
                    query_embeds_list.append(self.instance_embedding.weight[0:num_vec].to(dtype))
                else:
                    map_pos_embed = local_map_pos[i][valid_index]
                    map_label = local_map_input[i][valid_index]
                    map_label_embed = self.label_encoder(map_label)
                    map_query = torch.cat([map_pos_embed, map_label_embed], dim=-1)
                    query_embeds = self.instance_embedding.weight[0:num_vec].to(dtype)
                    query_embeds = self.query_attention(query_embeds.unsqueeze(0), key=map_query.unsqueeze(0))
                    query_embeds_list.append(query_embeds.squeeze(0))
            object_query_embeds = torch.stack(query_embeds_list, dim=0)
        
        if self.training and self.use_dino:
            object_query_embeds, self_attn_mask, max_gt_num_per_image, \
            gt_label_queries, gt_mask_queries, gt_box_queries = self.prepare_for_cdn(
                object_query_embeds=object_query_embeds,
                gt_bboxes_3d=gt_bboxes_3d, 
                gt_labels_3d=gt_labels_3d,
                num_group=self.num_group,
                label_noise_ratio=0.2,
                box_noise_scale=0.1,
                num_pts_per_vec=self.num_pts_per_vec,
                num_classes=self.cls_out_channels,
                hidden_dim=self.embed_dims,
                label_enc=self.label_encoder,
                box_enc=self.box_encoder,
            )
            num_vec += max_gt_num_per_image * (self.num_group + 1)
        # make attn mask
        # attention mask to prevent information leakage
        # self_attn_mask = (
        #     torch.zeros([num_vec, num_vec, ]).bool().to(mlvl_feats[0].device)
        # )
        # self_attn_mask[self.num_vec_one2one:, 0: self.num_vec_one2one, ] = True
        # self_attn_mask[0: self.num_vec_one2one, self.num_vec_one2one:, ] = True
        elif self.k_one2many and self.training:
            self_attn_mask = (
                torch.zeros([num_vec, num_vec,]).bool().to(mlvl_feats[0].device)
            )
            self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one,] = True
            self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :,] = True
        else:
            self_attn_mask = None

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                local_map=local_map,
            )['bev']
        else:
            # print("self_attn_mask: ", self_attn_mask)
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                local_map=local_map,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
                num_group=self.num_group,
            )

        bev_embed, depth, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        # outputs_classes_one2many = []
        # outputs_coords_one2many = []
        # outputs_pts_coords_one2many = []
        if self.training and self.k_one2many:
            outputs_classes_one2many = []
            outputs_coords_one2many = []
            outputs_pts_coords_one2many = []
        elif self.training and self.num_group:
            outputs_classes_group = {}
            outputs_coords_group = {}
            outputs_pts_coords_group = {}
            for i in range(self.num_group):
                outputs_classes_group[i] = []
                outputs_coords_group[i] = []
                outputs_pts_coords_group[i] = []
        
        if self.training and self.use_dino:
            outputs_classes_dn = []
            outputs_coords_dn = []
            outputs_pts_coords_dn = []
            outputs_classes_group_dn = {}
            outputs_coords_group_dn = {}
            outputs_pts_coords_group_dn = {}
            for i in range(self.num_group):
                outputs_classes_group_dn[i] = []
                outputs_coords_group_dn[i] = []
                outputs_pts_coords_group_dn[i] = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                if 'pts' not in self.query_embed_type:
                    reference = init_reference
                else:
                    reference = init_reference[..., 0:2] if not self.z_cfg['gt_z_flag'] else init_reference[..., 0:3]
            else:
                if 'pts' not in self.query_embed_type:
                    reference = inter_references[lvl - 1]
                else:
                    reference = inter_references[lvl - 1][..., 0:2] if not self.z_cfg['gt_z_flag'] else \
                    inter_references[lvl - 1][..., 0:3]
            reference = inverse_sigmoid(reference)
            # import pdb;pdb.set_trace()
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            if 'pts' not in self.query_embed_type:
                outputs_class = self.cls_branches[lvl](hs[lvl])
                tmp = self.reg_branches[lvl](hs[lvl])
                tmp[..., :] += reference[..., :]
                tmp = tmp.sigmoid()  # cx,cy,w,h
            else:
                outputs_class = self.cls_branches[lvl](hs[lvl]
                                                       .view(bs, num_vec, self.num_pts_per_vec, -1)
                                                       .mean(2))
                tmp = self.reg_branches[lvl](hs[lvl])
                tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
                # TODO: check the shape of reference
                # assert reference.shape[-1] == 2
                # tmp[..., 0:2] += reference[..., 0:2]
                # assert reference.shape[-1] == 2
                tmp += reference
                tmp = tmp.sigmoid()  # cx,cy,w,h
            # if not self.z_cfg['gt_z_flag']:
            # tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
            # TODO: check if using sigmoid
            outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, 0:self.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0:self.num_vec_one2one])

            if self.training and self.use_dino:
                outputs_classes_dn.append(outputs_class[:, self.num_vec_one2one:self.num_vec_one2one+max_gt_num_per_image])
                outputs_coords_dn.append(outputs_coord[:, self.num_vec_one2one:self.num_vec_one2one+max_gt_num_per_image])
                outputs_pts_coords_dn.append(outputs_pts_coord[:, self.num_vec_one2one:self.num_vec_one2one+max_gt_num_per_image])

            # outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
            # outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
            # outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])
            if self.k_one2many and self.training:
                outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
                outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
                outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])
            elif self.num_group and self.training:
                if self.use_dino:
                    num_vec_per_group = self.num_vec_one2one + max_gt_num_per_image
                    # print("outputs_class: ", outputs_class.shape)
                    # sys.exit()
                    for i in range(self.num_group):
                        outputs_classes_group[i].append( \
                            outputs_class[:, num_vec_per_group*(i+1):num_vec_per_group*(i+2)-max_gt_num_per_image])
                        outputs_coords_group[i].append( \
                            outputs_coord[:, num_vec_per_group*(i+1):num_vec_per_group*(i+2)-max_gt_num_per_image])
                        outputs_pts_coords_group[i].append( \
                            outputs_pts_coord[:, num_vec_per_group*(i+1):num_vec_per_group*(i+2)-max_gt_num_per_image])
                        outputs_classes_group_dn[i].append( \
                            outputs_class[:, num_vec_per_group*(i+1)+self.num_vec_one2one:num_vec_per_group*(i+2)])
                        outputs_coords_group_dn[i].append( \
                            outputs_coord[:, num_vec_per_group*(i+1)+self.num_vec_one2one:num_vec_per_group*(i+2)])
                        outputs_pts_coords_group_dn[i].append( \
                            outputs_pts_coord[:, num_vec_per_group*(i+1)+self.num_vec_one2one:num_vec_per_group*(i+2)])
                else:
                    for i in range(self.num_group):
                        outputs_classes_group[i].append( \
                            outputs_class[:, self.num_vec_one2one*(i+1):self.num_vec_one2one*(i+2)])
                        outputs_coords_group[i].append( \
                            outputs_coord[:, self.num_vec_one2one*(i+1):self.num_vec_one2one*(i+2)])
                        outputs_pts_coords_group[i].append( \
                            outputs_pts_coord[:, self.num_vec_one2one*(i+1):self.num_vec_one2one*(i+2)])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        # outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        # outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        # outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        if not self.training:
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes_one2one,
                'all_bbox_preds': outputs_coords_one2one,
                'all_pts_preds': outputs_pts_coords_one2one,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_pts_preds': None,
                'depth': depth,
                'seg': None,
                'pv_seg': None,
            }
            return outs

        if self.k_one2many:
            outputs_classes_one2many = torch.stack(outputs_classes_one2many)
            outputs_coords_one2many = torch.stack(outputs_coords_one2many)
            outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)
        elif self.num_group:
            group_outs = {}
            for i in range(self.num_group):
                group_outs[i] = {}
                group_outs[i]["enc_cls_scores"] = None
                group_outs[i]["enc_bbox_preds"] = None
                group_outs[i]["enc_pts_preds"] = None
                group_outs[i]["seg"] = None
                group_outs[i]["pv_seg"] = None
                group_outs[i]["all_cls_scores"] = torch.stack(outputs_classes_group[i])
                group_outs[i]["all_bbox_preds"] = torch.stack(outputs_coords_group[i])
                group_outs[i]["all_pts_preds"] = torch.stack(outputs_pts_coords_group[i])

        outputs_seg = None
        outputs_pv_seg = None
        if self.aux_seg['use_aux_seg']:
            seg_bev_embed = bev_embed.permute(1, 0, 2).view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            if self.aux_seg['bev_seg']:
                outputs_seg = self.seg_head(seg_bev_embed)
            bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
            if self.aux_seg['pv_seg']:
                outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

        # outs = {
        #     'bev_embed': bev_embed,
        #     'all_cls_scores': outputs_classes_one2one,
        #     'all_bbox_preds': outputs_coords_one2one,
        #     'all_pts_preds': outputs_pts_coords_one2one,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None,
        #     'enc_pts_preds': None,
        #     'depth': depth,
        #     'seg': outputs_seg,
        #     'pv_seg': outputs_pv_seg,
        #     "one2many_outs": dict(
        #         all_cls_scores=outputs_classes_one2many,
        #         all_bbox_preds=outputs_coords_one2many,
        #         all_pts_preds=outputs_pts_coords_one2many,
        #         enc_cls_scores=None,
        #         enc_bbox_preds=None,
        #         enc_pts_preds=None,
        #         seg=None,
        #         pv_seg=None,
        #     )
        # }
        if self.k_one2many:
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes_one2one,
                'all_bbox_preds': outputs_coords_one2one,
                'all_pts_preds': outputs_pts_coords_one2one,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_pts_preds': None,
                'depth': depth,
                'seg': outputs_seg,
                'pv_seg': outputs_pv_seg,
                "one2many_outs": dict(
                    all_cls_scores=outputs_classes_one2many,
                    all_bbox_preds=outputs_coords_one2many,
                    all_pts_preds=outputs_pts_coords_one2many,
                    enc_cls_scores=None,
                    enc_bbox_preds=None,
                    enc_pts_preds=None,
                    seg=None,
                    pv_seg=None,
                )
            }
        elif self.num_group:
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes_one2one,
                'all_bbox_preds': outputs_coords_one2one,
                'all_pts_preds': outputs_pts_coords_one2one,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_pts_preds': None,
                'depth': depth,
                'seg': outputs_seg,
                'pv_seg': outputs_pv_seg,
                "group_outs": group_outs,
            }
        
        if self.use_dino:
            # print("gt_label_queries: ", gt_label_queries.shape)
            outs['dn_cls_scores'] = torch.stack(outputs_classes_dn)
            outs['dn_bbox_preds'] = torch.stack(outputs_coords_dn)
            outs['dn_pts_preds'] = torch.stack(outputs_pts_coords_dn)
            outs['dn_gt_labels'] = gt_label_queries[:, :max_gt_num_per_image]
            outs['dn_gt_masks'] = gt_mask_queries[:, :max_gt_num_per_image]
            outs['dn_gt_pts'] = gt_box_queries[:, :max_gt_num_per_image]
            outs['num_dn_query'] = max_gt_num_per_image
            if self.num_group:
                for i in range(self.num_group):
                    outs['group_outs'][i]['dn_cls_scores'] = torch.stack(outputs_classes_group_dn[i])
                    outs['group_outs'][i]['dn_bbox_preds'] = torch.stack(outputs_coords_group_dn[i])
                    outs['group_outs'][i]['dn_pts_preds'] = torch.stack(outputs_pts_coords_group_dn[i])
                    outs['group_outs'][i]['dn_gt_labels'] = \
                        gt_label_queries[:, max_gt_num_per_image*(i+1):max_gt_num_per_image*(i+2)]
                    outs['group_outs'][i]['dn_gt_masks'] = \
                        gt_mask_queries[:, max_gt_num_per_image*(i+1):max_gt_num_per_image*(i+2)]
                    outs['group_outs'][i]['dn_gt_pts'] = \
                        gt_box_queries[:, max_gt_num_per_image*(i+1):max_gt_num_per_image*(i+2)]
                    outs['group_outs'][i]['num_dn_query'] = max_gt_num_per_image

        return outs
    
    def prepare_for_cdn(self, object_query_embeds, gt_bboxes_3d, gt_labels_3d, num_group, 
                        label_noise_ratio, box_noise_scale, num_pts_per_vec, num_classes, 
                        hidden_dim, label_enc, box_enc):
        device = gt_labels_3d[0].device
        gt_nums_per_image = [x.numel() * 2 for x in gt_labels_3d]
        max_gt_num_per_image = max(gt_nums_per_image)
        dn_group = num_group + 1
        batch_size = len(gt_labels_3d)
        
        # 以group為單位生成queries和gts
        gt_labels_list = []
        gt_boxes_list = []
        noisy_labels_list = []
        noisy_boxes_list = []
        for _ in range(dn_group):
            pos_labels = torch.cat(gt_labels_3d).view(-1)    # all gts
            neg_labels = torch.ones_like(pos_labels) * (num_classes)
            gt_labels = torch.cat([pos_labels, neg_labels], dim=0)
            gt_labels_list.append(copy.deepcopy(gt_labels))
            
            p = torch.rand_like(pos_labels.float())
            noised_index = torch.nonzero(p < label_noise_ratio).view(-1)
            new_labels = torch.randint_like(noised_index, 0, num_classes)
            noisy_pos_labels = copy.deepcopy(pos_labels.scatter(0, noised_index, new_labels))
            p = torch.rand_like(pos_labels.float())
            noised_index = torch.nonzero(p < label_noise_ratio).view(-1)
            new_labels = torch.randint_like(noised_index, 0, num_classes)
            noisy_neg_labels = copy.deepcopy(pos_labels.scatter(0, noised_index, new_labels))
            noisy_labels_list.append(torch.cat([noisy_pos_labels, noisy_neg_labels], dim=0))
            
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v2_dn.to(device) \
                                  for gt_bboxes in gt_bboxes_3d]
            gt_pts_list = [gt_shifts_pts[:, 0, :, :].contiguous().view(-1, self.code_size) \
                           for gt_shifts_pts in gt_shifts_pts_list]
            if self.code_size == 2:
                normalized_gt_pts_list = [normalize_2d_pts(gt_pts, self.pc_range) \
                                          for gt_pts in gt_pts_list]
            elif self.code_size == 3:
                normalized_gt_pts_list = [normalize_3d_pts(gt_pts, self.pc_range) \
                                          for gt_pts in gt_pts_list]
            else:
                raise NotImplementedError
            pos_pts = torch.cat(normalized_gt_pts_list).view(-1, num_pts_per_vec, self.code_size)
            neg_pts = copy.deepcopy(pos_pts)
            gt_boxes_list.append(copy.deepcopy(torch.cat([pos_pts, neg_pts], dim=0)))
            if self.code_size == 2:
                yx_scale = pos_pts[..., 1].max() / pos_pts[..., 0].max()
                pts_distance = torch.sqrt(((pos_pts[:, 0] - pos_pts[:, 1]) ** 2).sum(-1))
                rand_sign_pos = torch.randint_like(pos_pts, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
                rand_prob_pos = torch.rand_like(pos_pts) * rand_sign_pos * box_noise_scale
                diff_pos = (rand_prob_pos * pts_distance[:, None, None]) * torch.tensor((1, yx_scale)).to(device)[None]
                rand_sign_neg = torch.randint_like(pos_pts, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
                rand_prob_neg = torch.rand_like(pos_pts) * rand_sign_neg * box_noise_scale * 2
                diff_neg = (rand_prob_neg * pts_distance[:, None, None]) * torch.tensor((1, yx_scale)).to(device)[None]
            else:
                raise NotImplementedError
            noisy_pos_pts = (pos_pts + diff_pos).clamp(min=0.0, max=1.0)
            noisy_neg_pts = (neg_pts + diff_neg).clamp(min=0.0, max=1.0)
            noisy_boxes_list.append(torch.cat([noisy_pos_pts, noisy_neg_pts], dim=0))
        gt_labels = torch.cat(gt_labels_list, dim=0)
        gt_boxes = torch.cat(gt_boxes_list, dim=0)
        noised_labels = torch.cat(noisy_labels_list, dim=0)
        noised_boxes = torch.cat(noisy_boxes_list, dim=0)
        # noised_label_embedding = label_enc(noised_labels)
        noised_label_embedding = label_enc(F.one_hot(noised_labels.view(-1), num_classes=num_classes).float())
        # noised_box_embedding = box_enc(noised_boxes)
        # noised_object_embedding = noised_label_embedding.view(-1, 1, hidden_dim) + noised_box_embedding
        noised_object_embedding = noised_label_embedding.view(-1, 1, hidden_dim) + self.pts_embedding.weight.unsqueeze(0)
        
        # 以batch為單位生成queries
        noised_query_nums = max_gt_num_per_image * dn_group
        noised_object_queries = \
            torch.zeros(noised_query_nums, num_pts_per_vec, hidden_dim).to(device).repeat(batch_size, 1, 1, 1)
        batch_idx = torch.arange(0, batch_size)
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())
        batch_idx_per_group = batch_idx_per_instance.repeat(dn_group, 1).flatten()
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.tensor(list(range(num))) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([valid_index_per_group + max_gt_num_per_image * i for i in range(dn_group)]).long()
        if len(batch_idx_per_group):
            noised_object_queries[(batch_idx_per_group, valid_index_per_group)] = noised_object_embedding
        
        # 以batch為單位生成gts
        gt_masks = torch.ones_like(gt_labels).long()
        gt_label_queries = \
            torch.zeros(noised_query_nums, 1).to(device).repeat(batch_size, 1, 1).long()
        gt_mask_queries = \
            torch.zeros(noised_query_nums, 1).to(device).repeat(batch_size, 1, 1).long()
        gt_box_queries = \
            torch.zeros(noised_query_nums, num_pts_per_vec, self.code_size).to(device).repeat(batch_size, 1, 1, 1)
        gt_label_queries[(batch_idx_per_group, valid_index_per_group)] = gt_labels.view(-1, 1)
        gt_mask_queries[(batch_idx_per_group, valid_index_per_group)] = gt_masks.view(-1, 1)
        gt_box_queries[(batch_idx_per_group, valid_index_per_group)] = gt_boxes

        object_query_embeds = object_query_embeds.view(batch_size, dn_group, -1, num_pts_per_vec, hidden_dim)
        noised_object_queries = noised_object_queries.view(batch_size, dn_group, -1, num_pts_per_vec, hidden_dim)
        object_query_embeds = torch.cat([object_query_embeds, noised_object_queries], dim=2)
        # object_query_embeds = object_query_embeds.view(batch_size, -1, num_pts_per_vec, hidden_dim)
        object_query_embeds = object_query_embeds.view(batch_size, -1, hidden_dim)

        num_vec_per_group = object_query_embeds.shape[1] // dn_group // num_pts_per_vec
        self_attn_mask = torch.zeros([num_vec_per_group, num_vec_per_group]).bool().to(device)
        self_attn_mask[:-max_gt_num_per_image, :-max_gt_num_per_image] = True
        self_attn_mask[-max_gt_num_per_image:, -max_gt_num_per_image:] = True

        return object_query_embeds, self_attn_mask, max_gt_num_per_image, \
            gt_label_queries, gt_mask_queries, gt_box_queries

    def transform_box(self, pts, num_vec=50, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        if self.z_cfg['gt_z_flag']:
            pts_reshape = pts.view(pts.shape[0], num_vec,
                                   self.num_pts_per_vec, 3)
        else:
            pts_reshape = pts.view(pts.shape[0], num_vec,
                                   self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                                          gt_bboxes, gt_labels, gt_shifts_pts,
                                                          gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                                          pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                      :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else normalize_3d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                      align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :], normalized_pts_targets[isnotnan,
                                       :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else denormalize_3d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval:, :] - denormed_pts_preds[:,
                                                                                :-self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval:, :] - pts_targets[:, :-self.dir_interval, :]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :], pts_targets_dir[isnotnan,
                                                    :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_seg_mask,
             gt_pv_seg_mask,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds = preds_dicts['enc_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        if self.aux_seg['use_aux_seg']:
            # import ipdb;ipdb.set_trace()
            if self.aux_seg['bev_seg']:
                if preds_dicts['seg'] is not None:
                    seg_output = preds_dicts['seg']
                    num_imgs = seg_output.size(0)
                    seg_gt = torch.stack([gt_seg_mask[i] for i in range(num_imgs)], dim=0)
                    loss_seg = self.loss_seg(seg_output, seg_gt.float())
                    loss_dict['loss_seg'] = loss_seg
            if self.aux_seg['pv_seg']:
                # import ipdb;ipdb.set_trace()
                if preds_dicts['pv_seg'] is not None:
                    pv_seg_output = preds_dicts['pv_seg']
                    num_imgs = pv_seg_output.size(0)
                    pv_seg_gt = torch.stack([gt_pv_seg_mask[i] for i in range(num_imgs)], dim=0)
                    loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                    loss_dict['loss_pv_seg'] = loss_pv_seg
        
        if self.use_dino:
            dn_cls_pred = preds_dicts['dn_cls_scores']
            # print("dn_cls_pred: ", dn_cls_pred.shape)    # 6, 4, 28, 3
            # sys.exit()
            dn_pts_pred = preds_dicts['dn_pts_preds']
            dn_cls_gt = preds_dicts['dn_gt_labels'].view(-1)
            dn_mask_gt = preds_dicts['dn_gt_masks']
            dn_pts_gt = preds_dicts['dn_gt_pts'].contiguous().view(-1, self.code_size)
            mask_cls = copy.deepcopy(dn_mask_gt.view(-1))
            mask_pts = copy.deepcopy(dn_mask_gt.view(-1))
            mask_pts[dn_cls_gt == self.num_cls_fcs] = 0
            mask_pts = mask_pts.view(-1, 1, 1)
            mask_dirs = copy.deepcopy(mask_pts)
            mask_dirs = mask_dirs.view(-1, 1)
            
            mask_pts = mask_pts.repeat(1, self.num_pts_per_vec, self.code_size).view(-1, self.code_size)
            if self.code_size == 2:
                dn_pts_pred_denorm = denormalize_2d_pts(dn_pts_pred, self.pc_range)
                dn_pts_gt_denorm = denormalize_2d_pts(preds_dicts['dn_gt_pts'], self.pc_range)
            elif self.code_size == 3:
                dn_pts_pred_denorm = denormalize_3d_pts(dn_pts_pred, self.pc_range)
                dn_pts_gt_denorm = denormalize_3d_pts(preds_dicts['dn_gt_pts'], self.pc_range)
            else:
                raise NotImplementedError
            dn_dirs_pred = dn_pts_pred_denorm[..., 1:, :] - dn_pts_pred_denorm[..., :-1, :]
            dn_dirs_gt = dn_pts_gt_denorm[..., 1:, :] - dn_pts_gt_denorm[..., :-1, :]
            dn_dirs_gt = dn_dirs_gt.view(-1, self.num_pts_per_vec - 1, self.code_size)
            mask_dirs = mask_dirs.repeat(1, self.num_pts_per_vec - 1)
            for lvl in range(dn_cls_pred.shape[0]):
                loss_dict[f'd{lvl}.dn.loss_cls'] = \
                    2.0 * self.loss_cls(dn_cls_pred[lvl].view(-1, self.cls_out_channels), \
                                        dn_cls_gt.long(), mask_cls, avg_factor=mask_cls.sum().item())
                loss_dict[f'd{lvl}.dn.loss_pts'] = \
                    5.0 * self.loss_pts(dn_pts_pred[lvl].view(-1, self.code_size), \
                                        dn_pts_gt, mask_pts, avg_factor=mask_pts.sum().item())
                loss_dict[f'd{lvl}.dn.loss_dirs'] = \
                    0.005 * self.loss_dir(dn_dirs_pred[lvl].view(-1, self.num_pts_per_vec - 1, self.code_size), \
                                        dn_dirs_gt, mask_dirs, avg_factor=mask_dirs.sum().item())

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # TODO bug here
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_pts_preds,
                                 gt_bboxes_list, binary_labels_list, gt_pts_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_losses_iou'] = enc_losses_iou
            loss_dict['enc_losses_pts'] = enc_losses_pts
            loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                                                               losses_bbox[:-1],
                                                                               losses_iou[:-1],
                                                                               losses_pts[:-1],
                                                                               losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict
    
    @force_fp32(apply_to=('list_preds_dicts'))
    def loss_group(self, gt_bboxes_list, gt_labels_list, list_preds_dicts):

        # print("list_preds_dicts: ", len(list_preds_dicts), list_preds_dicts[0].keys())
        # sys.exit()

        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        
        all_cls_scores = []
        all_bbox_preds = []
        all_pts_preds = []
        for preds_dicts in list_preds_dicts:
            all_cls_scores.append(preds_dicts['all_cls_scores'])
            all_bbox_preds.append(preds_dicts['all_bbox_preds'])
            all_pts_preds.append(preds_dicts['all_pts_preds'])
        # print("before stack: ", all_cls_scores[0].shape)
        all_cls_scores = torch.stack(all_cls_scores).flatten(0,1)
        # print("after stack: ", all_cls_scores.shape)
        all_bbox_preds = torch.stack(all_bbox_preds).flatten(0,1)
        all_pts_preds = torch.stack(all_pts_preds).flatten(0,1)
            
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [None for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        count_dec_layer = 0
        count_group = 0
        num_dec_layers = num_dec_layers // self.num_group
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(losses_cls, losses_pts, losses_dir):
            loss_dict[f'g{count_group}.d{count_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'g{count_group}.d{count_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'g{count_group}.d{count_dec_layer}.loss_dir'] = loss_dir_i
            count_dec_layer += 1
            if count_dec_layer == num_dec_layers:
                count_dec_layer = 0
                count_group += 1
        
        if self.use_dino:
            for idx_group, preds_dicts in enumerate(list_preds_dicts):
                dn_cls_pred = preds_dicts['dn_cls_scores']
                # print("dn_cls_pred: ", idx_group, dn_cls_pred.shape)
                # sys.exit()
                dn_pts_pred = preds_dicts['dn_pts_preds']
                dn_cls_gt = preds_dicts['dn_gt_labels'].view(-1)
                dn_mask_gt = preds_dicts['dn_gt_masks']
                dn_pts_gt = preds_dicts['dn_gt_pts'].contiguous().view(-1, self.code_size)
                mask_cls = copy.deepcopy(dn_mask_gt.view(-1))
                mask_pts = copy.deepcopy(dn_mask_gt.view(-1))
                mask_pts[dn_cls_gt == self.num_cls_fcs] = 0
                mask_pts = mask_pts.view(-1, 1, 1)
                mask_dirs = copy.deepcopy(mask_pts)
                mask_dirs = mask_dirs.view(-1, 1)
                mask_pts = mask_pts.repeat(1, self.num_pts_per_vec, self.code_size).view(-1, self.code_size)
                if self.code_size == 2:
                    dn_pts_pred_denorm = denormalize_2d_pts(dn_pts_pred, self.pc_range)
                    dn_pts_gt_denorm = denormalize_2d_pts(preds_dicts['dn_gt_pts'], self.pc_range)
                elif self.code_size == 3:
                    dn_pts_pred_denorm = denormalize_3d_pts(dn_pts_pred, self.pc_range)
                    dn_pts_gt_denorm = denormalize_3d_pts(preds_dicts['dn_gt_pts'], self.pc_range)
                else:
                    raise NotImplementedError
                dn_dirs_pred = dn_pts_pred_denorm[..., 1:, :] - dn_pts_pred_denorm[..., :-1, :]
                dn_dirs_gt = dn_pts_gt_denorm[..., 1:, :] - dn_pts_gt_denorm[..., :-1, :]
                dn_dirs_gt = dn_dirs_gt.view(-1, self.num_pts_per_vec - 1, self.code_size)
                mask_dirs = mask_dirs.repeat(1, self.num_pts_per_vec - 1)
                for lvl in range(dn_cls_pred.shape[0]):
                    # print("dn_cls_pred[lvl]: ", lvl, dn_cls_pred[lvl].shape)
                    # print("dn_cls_gt: ", dn_cls_gt.shape)
                    loss_dict[f'g{idx_group}.d{lvl}.dn.loss_cls'] = \
                        2.0 * self.loss_cls(dn_cls_pred[lvl].view(-1, self.cls_out_channels), \
                                            dn_cls_gt.long(), mask_cls, avg_factor=mask_cls.sum().item())
                    loss_dict[f'g{idx_group}.d{lvl}.dn.loss_pts'] = \
                        5.0 * self.loss_pts(dn_pts_pred[lvl].view(-1, self.code_size), \
                                            dn_pts_gt, mask_pts, avg_factor=mask_pts.sum().item())
                    loss_dict[f'g{idx_group}.d{lvl}.dn.loss_dirs'] = \
                        0.005 * self.loss_dir(dn_dirs_pred[lvl].view(-1, self.num_pts_per_vec - 1, self.code_size), \
                                            dn_dirs_gt, mask_dirs, avg_factor=mask_dirs.sum().item())
            
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

    @force_fp32(apply_to=('preds_dicts'))
    def get_pred_mask(self, preds_dicts, global_map_type='raster', status='train'):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        with torch.no_grad():
            raster_lists = self.bbox_coder.decode_raster(preds_dicts, status)
            # print("raster_lists: ", raster_lists)
            if global_map_type == 'raster':
                raster_tensor = torch.stack(raster_lists, dim=0)
                # bs, n, bev_h, bev_w = raster_tensor.shape
                raster_tensor = raster_tensor.permute(0, 2, 3, 1)
            elif global_map_type.startswith('vector'):
                return raster_lists
            else:
                raise NotImplementedError("not supported global map type")
            return raster_tensor