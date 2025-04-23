
import mmcv
import os
import json
from shapely.geometry import LineString
import numpy as np


def timestamp_indexed(dict_gt_pkl):
    num_class = 3
    CLASS2LABEL_GLOBAL = {
            'divider': 0,
            'ped_crossing': 1,
            'boundary': 2,
            'others': -1
        }
    dict_gt_hdmap = {}
    for sample in dict_gt_pkl['infos']:
        city_name = sample['map_location']
        if city_name not in dict_gt_hdmap.keys():
            dict_gt_hdmap[city_name] = {}
        dict_gt_hdmap[city_name][str(sample['timestamp'])] = {}
        for idx_class in range(num_class):
            dict_gt_hdmap[city_name][str(sample['timestamp'])][str(idx_class)] = []
        for name, list_vec in sample['annotation'].items():
            if name in CLASS2LABEL_GLOBAL.keys():
                idx_class = str(CLASS2LABEL_GLOBAL[name])
                for vec in list_vec:
                    vec = LineString(vec)
                    distances = np.linspace(0, vec.length, 20)
                    vec_itpl = np.array([list(vec.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
                    dict_gt_hdmap[city_name][str(sample['timestamp'])][str(idx_class)].append(vec_itpl.tolist())
    return dict_gt_hdmap


dir_gt_pkl_train = "./data/nuscenes/nuscenes_map_infos_temporal_train.pkl"
dict_gt_pkl_train = mmcv.load(dir_gt_pkl_train)
dict_gt_hdmap_train = timestamp_indexed(dict_gt_pkl_train)
dir_gt_pkl_val = "./data/nuscenes/nuscenes_map_infos_temporal_val.pkl"
dict_gt_pkl_val = mmcv.load(dir_gt_pkl_val)
dict_gt_hdmap_val = timestamp_indexed(dict_gt_pkl_val)

for city_name in dict_gt_hdmap_val.keys():
    for timestamp in dict_gt_hdmap_val[city_name].keys():
        dict_gt_hdmap_train[city_name][timestamp] = dict_gt_hdmap_val[city_name][timestamp]

dir_trainval = "./data/nuscenes/nuscenes_map_infos_temporal_trainval.json"
with open(dir_trainval, 'w') as f:
    json.dump(dict_gt_hdmap_train, f)
f.close()
