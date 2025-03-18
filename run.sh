source /baai-cwm-1/baai_cwm_ml/algorithm/chongjie.ye/envs/miniconda3/etc/profile.d/conda.sh 
conda activate ovmono3d
cd /baai-cwm-1/baai_cwm_ml/algorithm/chongjie.ye/code/ovmono3d
nohup python tools/train_net.py \
    --config-file configs/OVMono3D_dinov2_SFP.yaml \
    --num-gpus 2 \
    OUTPUT_DIR /baai-cwm-1/baai_cwm_ml/algorithm/chongjie.ye/output/ovmono3d_depth \
    VIS_PERIOD 10000 \
    TEST.EVAL_PERIOD 10000 \
    MODEL.STABILIZE 0.03 \
    SOLVER.BASE_LR 0.012 \
    SOLVER.CHECKPOINT_PERIOD 9999 \
    SOLVER.IMS_PER_BATCH 8 \
    > nohup.out 2>&1 &