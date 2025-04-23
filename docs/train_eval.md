# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test
# Uni-PrevPredMap is exclusively trained using a 4-GPU configuration. For 8-GPU deployments, setting the map_multi parameter to 0 can be experimentally attempted.

Train Uni-PrevPredMap with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/uniprevpredmap/uniprevpredmap_nusc_r50_24ep.py 4
```

Eval Uni-PrevPredMap with 4 GPUs
```
./tools/dist_test_map.sh ./projects/configs/uniprevpredmap/uniprevpredmap_nusc_r50_24ep.py ./path/to/ckpts.pth 4
```
