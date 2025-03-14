# Copyright (c) Meta Platforms, Inc. and affiliates
import itertools
import logging
import numpy as np
import math
import json
from collections import defaultdict
import torch
import torch.utils.data

from detectron2.config import configurable
from detectron2.utils.logger import _log_api_usage

from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler, 
    RepeatFactorTrainingSampler, 
    TrainingSampler
)
from detectron2.data.build import (
    filter_images_with_only_crowd_annotations, 
    build_batch_data_loader,
    trivial_batch_collator
)
import random


def sample_by_percentage(data_list, percentage, seed=None):
    if seed is not None:
        random.seed(seed)  
    sample_size = int(len(data_list) * percentage)
    return random.sample(data_list, sample_size)

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    return [x_min, y_min, x_max, y_max]


def merge_oracle2d_to_detection_dicts(dataset_dicts, oracle2d):
    for dataset, oracle in zip(dataset_dicts, oracle2d):
        with open(oracle, 'r') as file:
            oracle_data = json.load(file)
        for data_dict, oracle_dict in zip(dataset,oracle_data):
            assert data_dict['image_id'] == oracle_dict['image_id']
            data_dict["oracle2D"] = {"gt_bbox2D": torch.tensor([xywh_to_xyxy(instance["bbox"]) for instance in oracle_dict["instances"]]), 
                                     "gt_classes": torch.tensor([instance["category_id"] for instance in oracle_dict["instances"]]),
                                     "gt_scores": torch.tensor([instance["score"] for instance in oracle_dict["instances"]]),
                                     }


def get_detection_dataset_dicts(names, filter_empty=True, oracle2d=None, **kwargs):
    
    if isinstance(names, str):
        names = [names]

    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    if oracle2d:
        merge_oracle2d_to_detection_dicts(dataset_dicts, oracle2d)
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None, dataset_id_to_src=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if cfg.INPUT.TRAIN_SET_PERCENTAGE != 1.0:
        dataset = sample_by_percentage(dataset, cfg.INPUT.TRAIN_SET_PERCENTAGE, seed=42) 
    logger = logging.getLogger(__name__)
    logger.info("Using {} training images".format(len(dataset)))
    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        balance_datasets = cfg.DATALOADER.BALANCE_DATASETS
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))

        if balance_datasets:
            assert dataset_id_to_src is not None, 'Need dataset sources.'

            dataset_source_to_int = {val:i for i, val in enumerate(set(dataset_id_to_src.values()))}
            dataset_ids_per_img = [dataset_source_to_int[dataset_id_to_src[img['dataset_id']]] for img in dataset]
            dataset_ids = np.unique(dataset_ids_per_img)

            # only one source? don't re-weight then.
            if len(dataset_ids) == 1:
                weights_per_img = torch.ones(len(dataset_ids_per_img)).float()
            
            # compute per-dataset weights.
            else:
                counts = np.bincount(dataset_ids_per_img)
                counts = [counts[id] for id in dataset_ids]
                weights = [1 - count/np.sum(counts) for count in counts]
                weights = [weight/np.min(weights) for weight in weights]
                
                weights_per_img = torch.zeros(len(dataset_ids_per_img)).float()
                dataset_ids_per_img = torch.FloatTensor(dataset_ids_per_img).long()

                # copy weights
                for dataset_id, weight in zip(dataset_ids, weights):
                    weights_per_img[dataset_ids_per_img == dataset_id] = weight

        # no special sampling whatsoever
        if sampler_name == "TrainingSampler" and not balance_datasets:
            sampler = TrainingSampler(len(dataset))

        # balance the weight sampling by datasets
        elif sampler_name == "TrainingSampler" and balance_datasets:
            sampler = RepeatFactorTrainingSampler(weights_per_img)
        
        # balance the weight sampling by categories
        elif sampler_name == "RepeatFactorTrainingSampler" and not balance_datasets:
            repeat_factors = repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)

        # balance the weight sampling by categories AND by dataset frequency
        elif sampler_name == "RepeatFactorTrainingSampler" and balance_datasets:
            repeat_factors = repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            repeat_factors *= weights_per_img
            repeat_factors /= repeat_factors.min().item()
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.

        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.

        Returns:
            torch.Tensor:
                the i-th element is the repeat factor for the dataset image at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                if cat_id < 0: continue
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids if cat_id >= 0}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)

    def collate_fn(batch):
        """
        Custom collate function to handle both images and depth maps
        """
        batched_inputs = []
        for per_image in batch:

            data = {
                "image": per_image["image"],
                "height": per_image["height"],
                "width": per_image["width"],
            }

            if "depth" in per_image:
                data["depth"] = per_image["depth"]
            
            if "instances" in per_image:
                data["instances"] = per_image["instances"]
            if "dataset_id" in per_image:
                data["dataset_id"] = per_image["dataset_id"]
            if "K" in per_image:
                data["K"] = per_image["K"]
                
            batched_inputs.append(data)
            
        return batched_inputs

    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def _test_loader_from_config(cfg, dataset_name, mode, mapper=None):
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        oracle2d=[
            getattr(getattr(cfg.DATASETS.ORACLE2D_FILES[cfg.DATASETS.ORACLE2D_FILES.EVAL_MODE], mode), x) for x in dataset_name
        ]
        if cfg.TEST.ORACLE2D
        else None,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper(cfg, False)

    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS}

def test_data_collate_fn(batch):
    """
    Custom collate function for testing
    """
    batched_inputs = []
    for per_image in batch:
        data = {
            "image": per_image["image"],
            "height": per_image["height"],
            "width": per_image["width"],
        }
        
        # Add image_id if it exists
        if "image_id" in per_image:
            data["image_id"] = per_image["image_id"]
        
        if "depth" in per_image:
            data["depth"] = per_image["depth"]
        else:
            print("No depth data found in this sample")

        if "file_name" in per_image:
            data["file_name"] = per_image["file_name"]
        if "dataset_id" in per_image:
            data["dataset_id"] = per_image["dataset_id"]
        if "K" in per_image:
            data["K"] = per_image["K"]
            
        batched_inputs.append(data)
        
    return batched_inputs

@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=test_data_collate_fn,
    )
    return data_loader