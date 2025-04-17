# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import torch
import numpy as np
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import (
    DatasetMapper
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)
import os

class DatasetMapper3D(DatasetMapper):
    def __init__(self, cfg, is_train = True):
        super().__init__(cfg, is_train)
        self.depth_dir = "/baai-cwm-nas/algorithm/chongjie.ye/data/OmniNOCS/ARKitScenes/ARKitScenes_nocs"
        self.use_depth = cfg.MODEL.FPN.USE_DEPTH_FUSION
        self.nocs_dir = "/baai-cwm-nas/algorithm/chongjie.ye/data/OmniNOCS/objectron_data/objectron_nocs"
        self.use_nocs = cfg.MODEL.FPN.USE_NOCS_FUSION 

    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        image = detection_utils.read_image(dataset_dict["file_path"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)


        if self.use_depth:
            depth_path = os.path.join(self.depth_dir, dataset_dict["file_name"]+ '.npy')
            
            try:
                if os.path.exists(depth_path):
                    # 直接加载npy文件
                    depth_data = np.load(depth_path)
                   
                    if len(depth_data.shape) > 2:
                        # 检查是否为5维数据 [1, 1, H, W, 1]
                        if len(depth_data.shape) == 5:
                            depth_data = depth_data[0, 0, :, :, 0]  
                        elif len(depth_data.shape) == 3 and depth_data.shape[2] == 1:
                            depth_data = depth_data[:, :, 0]  
                    
                    depth = torch.as_tensor(depth_data.astype("float32"))
                    
                    # match depth size to image size
                    if depth.shape[:2] != image.shape[:2]:
                        depth = torch.nn.functional.interpolate(
                            depth.unsqueeze(0).unsqueeze(0),  # 确保是[N,C,H,W]格式
                            size=image.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                else:
                    print(f"Depth file does not exist: {depth_path}")
                    depth = torch.zeros(image.shape[:2], dtype=torch.float32)
            except Exception as e:
                print(f"Error reading depth file: {depth_path}, Error: {str(e)}")
                depth = torch.zeros(image.shape[:2], dtype=torch.float32)
        else:
            depth = None

        # Load NOCS map from PNG image
        if self.use_nocs:
            nocs_path = os.path.join(self.nocs_dir, dataset_dict["file_name"] + '_nocs.png')
            try:
                nocs_image = detection_utils.read_image(nocs_path, format="RGB")
                nocs = torch.as_tensor(nocs_image.astype("float32") / 255.0)  # Normalize to [0,1]
                
                # match nocs size to image size
                if nocs.shape[:2] != image.shape[:2]:
                    nocs = torch.nn.functional.interpolate(
                        nocs.permute(2, 0, 1).unsqueeze(0),
                        size=image.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
            except Exception as e:
                print(f"Error reading NOCS file: {nocs_path}, Error: {str(e)}")
                nocs = torch.zeros((*image.shape[:2], 3), dtype=torch.float32)
        else:
            nocs = None

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if depth is not None:
            depth = transforms.apply_image(depth.numpy())
            depth = np.ascontiguousarray(depth) 
            depth = torch.as_tensor(depth)
            dataset_dict["depth"] = depth.unsqueeze(0) 

        # Apply transforms to NOCS map
        if nocs is not None:
            nocs = transforms.apply_image(nocs.numpy())
            nocs = np.ascontiguousarray(nocs)
            nocs = torch.as_tensor(nocs)
            dataset_dict["nocs"] = nocs.permute(2, 0, 1)  # Convert to CxHxW format

        # no need for additoinal processing at inference
        if not self.is_train:
            return dataset_dict

        if "annotations" in dataset_dict:

            dataset_id = dataset_dict['dataset_id']
            K = np.array(dataset_dict['K'])

            unknown_categories = self.dataset_id_to_unknown_cats[dataset_id]

            # transform and pop off annotations
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]

            # convert to instance format
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict

'''
Cached for mirroring annotations
'''
_M1 = np.array([
    [1, 0, 0], 
    [0, -1, 0],
    [0, 0, -1]
])
_M2 = np.array([
    [-1.,  0.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0.,  1.]
])


def transform_instance_annotations(annotation, transforms, *, K):
    
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))[0]
    
    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if annotation['center_cam'][2] != 0:

        # project the 3D box annotation XYZ_3D to screen 
        point3D = annotation['center_cam']
        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        # apply coords transforms to 2D box
        annotation["center_cam_proj"][0:2] = transforms.apply_coords(
            point2D[np.newaxis][:, :2]
        )[0].tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]
        
        if annotation['ignore']:
            # all keypoints marked as not visible 
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:
            
            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        # in place
        transforms.apply_coords(keypoints[:, :2])
        annotation["keypoints"] = keypoints.tolist()

        # manually apply mirror for pose
        for transform in transforms:

            # horrizontal flip?
            if isinstance(transform, T.HFlipTransform):

                pose = _M1 @ np.array(annotation["pose"]) @ _M2
                annotation["pose"] = pose.tolist()
                annotation["R_cam"] = pose.tolist()

    return annotation


def annotations_to_instances(annos, image_size, unknown_categories):

    # init
    target = Instances(image_size)
    
    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor([int(obj["category_id"]) for obj in annos], dtype=torch.int64)
    target.gt_boxes = Boxes([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos])
    target.gt_boxes3D = torch.FloatTensor([anno['center_cam_proj'] + anno['dimensions'] + anno['center_cam'] for anno in annos])
    target.gt_poses = torch.FloatTensor([anno['pose'] for anno in annos])
    
    n = len(target.gt_classes)

    # do keypoints?
    target.gt_keypoints = Keypoints(torch.FloatTensor([anno['keypoints'] for anno in annos]))

    gt_unknown_category_mask = torch.zeros(max(unknown_categories)+1, dtype=bool)
    gt_unknown_category_mask[torch.tensor(list(unknown_categories))] = True

    # include available category indices as tensor with GTs
    target.gt_unknown_category_mask = gt_unknown_category_mask.unsqueeze(0).repeat([n, 1])

    return target
