set -x

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

proj_path=$(pwd)
dir_name=$(dirname "$0")
export PYTHONPATH=$proj_path:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $dir_name/test.py \
    --cfg_path=projects/configs/citystreet/swin_benchmark_lr_strong_aug.py \
    --chkp_path=workdirs/swin/[2023-07-28::03:40:00]/checkpoints/365.pth