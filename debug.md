run command:
```
nohup python tools/train_net.py  --eval-only --config-file configs/OVMono3D_dinov2_SFP.yaml --num-gpus 2 OUTPUT_DIR ../../output/ovmono3d_depth/ MODEL.WEIGHTS ../../output/ovmono3d_depth/model_recent.pth TEST.CAT_MODE "base" DATASETS.ORACLE2D_FILES.EVAL_MODE "target_aware" > nohup.out 2>&1 &
```
run on public model checkpoint:
```
nohup python tools/train_net.py  --eval-only --config-file configs/OVMono3D_dinov2_SFP.yaml --num-gpus 2 OUTPUT_DIR ../../output/ovmono3d_depth/ MODEL.WEIGHTS ./checkpoints/ovmono3d_lift.pth TEST.CAT_MODE "base" DATASETS.ORACLE2D_FILES.EVAL_MODE "target_aware" > nohup.out 2>&1 &
```

issue 1:origin inference result is wrong\
issue 2:AP2D abnormally high && AP3D abnormally low

step 1:try eval on public model checkpoint\
step 2:fix inference error to get true inference result\
step 3:try eval on public model checkpoint



error 1:when inference with debug branch, the depth don't input to the model\
when i fix error 1, test eval on public model checkpoint and own model checkpoint to see if the inference result is right\
error 2: