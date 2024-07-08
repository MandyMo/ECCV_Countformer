set -x

#solve the over-load problem
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

proj_path=$(pwd)
dir_name=$(dirname "$0")
export PYTHONPATH=$proj_path:$PYTHONPATH

# configs for each dataset
citystreet_cfg=projects/configs/citystreet/swin_benchmark_lr.py
cross_view_cfg=projects/configs/cross_view/voxel/swin_benchmark_lr_cam_embed_img_vox.py

#some parameters
seed=42
apply_image_mask=0
deter=0
amp=0

#resume or not
resume=0
only_weight=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $dir_name/train.py \
    --seed=$seed \
    --apply_image_mask=$apply_image_mask \
    --deter=$deter \
    --cfg_path=$cross_view_cfg \
    --resume=$resume \
    --chkp_path=$chkp_path \
    --only_weight=$only_weight \
    --amp=$amp