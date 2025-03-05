<div align="center">

<!-- # OVMono3D -->

# Open Vocabulary Monocular 3D Object Detection

[Jin Yao][jy], [Hao Gu][hg], [Xuweiyi Chen][xc], [Jiayun Wang][jw], [Zezhou Cheng][zc]


[![Website](https://img.shields.io/badge/Project-Page-b361ff
)](https://uva-computer-vision-lab.github.io/ovmono3d/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2411.16833)


</div>

<table style="border-collapse: collapse; border: none;">
<tr>
	<!-- <td width="60%">
		<p align="center">
			Zero-shot (+ tracking) on <a href="https://about.facebook.com/realitylabs/projectaria">Project Aria</a> data
			<img src=".github/generalization_demo.gif" alt="Aria demo video"/ height="300">
		</p>
	</td> -->
	<td width="100%">
		<p align="center">
			Zero-shot predictions on COCO
			<img src=".github/coco.png" alt="COCO demo"/ height="300">
		</p>
	</td>
</tr>
</table>


## Installation <a name="installation"></a>
Our used cuda version is 12.1.1.
Run
```bash
conda create -n ovmono3d python=3.8.20
conda activate ovmono3d

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```
to create the environment and install pytorch.

Run
```bash
sh setup.sh
```
to install additional dependencies and download model checkpoints of OVMono3D-LIFT and other foundation models.

## Demo <a name="demo"></a>
Run
```bash
python demo/demo.py --config-file configs/OVMono3D_dinov2_SFP.yaml \
	--input-folder datasets/coco_examples \
	--labels-file datasets/coco_examples/labels.json \
	--threshold 0.45 \
	MODEL.ROI_HEADS.NAME ROIHeads3DGDINO \
	MODEL.WEIGHTS checkpoints/ovmono3d_lift.pth \
	OUTPUT_DIR output/coco_examples 
```
to get the results for the example COCO images.

You can also try your own images and prompted category labels. See the format of the label file in [`labels.json`](datasets/coco_examples/labels.json). If you know the camera intrinsics you could input them as arguments with the convention `--focal-length <float>` and `--principal-point <float> <float>`. Check [`demo.py`](demo/demo.py) for more details.


## Data <a name="data"></a>
Please follow the instructions in [Omni3D](https://github.com/facebookresearch/omni3d/blob/main/DATA.md) to set up the datasets.  
Run
```bash
sh ./download_data.sh
```
to download our pre-processed OVMono3D 2D predictions (12 GB after unzipping).  


## Evaluation <a name="evaluation"></a>


To run inference and evaluation of OVMono3D-LIFT, use the following command:
```bash
python tools/train_net.py --eval-only  --config-file configs/OVMono3D_dinov2_SFP.yaml --num-gpus 2 \
    OUTPUT_DIR  output/ovmono3d_lift  \
    MODEL.WEIGHTS checkpoints/ovmono3d_lift.pth \
    TEST.CAT_MODE "novel" \
    DATASETS.ORACLE2D_FILES.EVAL_MODE "target_aware"
```
TEST.CAT_MODE denotes the category set to be evaluated: `novel` or `base` or `all`

DATASETS.ORACLE2D_FILES.EVAL_MODE denotes the evaluation protocol: `target_aware` or `previous_metric`

To run inference and evaluation of OVMono3D-GEO, use the following commands:
```bash
python tools/ovmono3d_geo.py
python tools/eval_ovmono3d_geo.py
```


## Training <a name="training"></a>

To run training of OVMono3D-LIFT, use the following command:
```bash
nohup python tools/train_net.py  --config-file configs/OVMono3D_dinov2_SFP.yaml     --num-gpus 2     OUTPUT_DIR ../../output/ovmono3d_depth     VIS_PERIOD 10000     TEST.EVAL_PERIOD 10000    MODEL.STABILIZE 0.03     SOLVER.BASE_LR 0.012     SOLVER.CHECKPOINT_PERIOD 10000     SOLVER.IMS_PER_BATCH 16     > nohup.out 2>&1 &
```

The training hyperparameters above are used in our experiments. While these parameters can be customized to suit your specific requirements, please note that performance may vary across different configurations.


## Citing <a name="citing"></a>
If you find this work useful for your research, please kindly cite:

```BibTeX
@article{yao2024open,
  title={Open Vocabulary Monocular 3D Object Detection},
  author={Yao, Jin and Gu, Hao and Chen, Xuweiyi and Wang, Jiayun and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2411.16833},
  year={2024}
}
```
Please also consider cite the awesome work of [Omni3D](https://github.com/facebookresearch/omni3d) and datasets used in Omni3D.
<details><summary>BibTex</summary>

```BibTeX
@inproceedings{brazil2023omni3d,
  author =       {Garrick Brazil and Abhinav Kumar and Julian Straub and Nikhila Ravi and Justin Johnson and Georgia Gkioxari},
  title =        {{Omni3D}: A Large Benchmark and Model for {3D} Object Detection in the Wild},
  booktitle =    {CVPR},
  address =      {Vancouver, Canada},
  month =        {June},
  year =         {2023},
  organization = {IEEE},
}
```

```BibTex
@inproceedings{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}
``` 

```BibTex
@inproceedings{caesar2020nuscenes,
  title={nuscenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```

```BibTex
@inproceedings{song2015sun,
  title={Sun rgb-d: A rgb-d scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}
```

```BibTex
@inproceedings{dehghan2021arkitscenes,
  title={{ARK}itScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB}-D Data},
  author={Gilad Baruch and Zhuoyuan Chen and Afshin Dehghan and Tal Dimry and Yuri Feigin and Peter Fu and Thomas Gebauer and Brandon Joffe and Daniel Kurz and Arik Schwartz and Elad Shulman},
  booktitle={NeurIPS Datasets and Benchmarks Track (Round 1)},
  year={2021},
}
```

```BibTex
@inproceedings{hypersim,
  author    = {Mike Roberts AND Jason Ramapuram AND Anurag Ranjan AND Atulit Kumar AND
                 Miguel Angel Bautista AND Nathan Paczan AND Russ Webb AND Joshua M. Susskind},
  title     = {{Hypersim}: {A} Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
  booktitle = {ICCV},
  year      = {2021},
}
```

```BibTex
@article{objectron2021,
  title={Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations},
  author={Ahmadyan, Adel and Zhang, Liangkai and Ablavatski, Artsiom and Wei, Jianing and Grundmann, Matthias},
  journal={CVPR},
  year={2021},
}
```

</details>


[jy]: https://yaojin17.github.io
[hg]: https://www.linkedin.com/in/hao--gu/
[xc]: https://xuweiyichen.github.io/
[jw]: https://pwang.pw/
[zc]: https://sites.google.com/site/zezhoucheng/

