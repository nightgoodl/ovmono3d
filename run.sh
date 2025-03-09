cd /baai-cwm-1/baai_cwm_ml/algorithm/chongjie.ye/code/ovmono3d 

mkdir -p ../../output/ovmono3d_depth

echo "Starting training..."

nohup python tools/train_net.py \
    --config-file configs/OVMono3D_dinov2_SFP.yaml \
    --num-gpus 8 \
    OUTPUT_DIR ../../output/ovmono3d_depth \
    VIS_PERIOD 10000 \
    TEST.EVAL_PERIOD 10000 \
    MODEL.STABILIZE 0.03 \
    SOLVER.BASE_LR 0.012 \
    SOLVER.CHECKPOINT_PERIOD 9999 \
    SOLVER.IMS_PER_BATCH 64 \
    > nohup.out 2>&1 &