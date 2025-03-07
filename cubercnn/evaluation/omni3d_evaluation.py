# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import copy
import datetime
import io
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from typing import List, Union
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as maskUtils
import torch
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table, log_every_n_seconds
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from detectron2.utils.comm import get_world_size, is_main_process
import detectron2.utils.comm as comm
from detectron2.evaluation import (
    DatasetEvaluators, inference_context, DatasetEvaluator
)
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from torch import nn

import logging
from cubercnn.data import Omni3D
from pytorch3d import _C
import torch.nn.functional as F

from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles

import cubercnn.vis.logperf as utils_logperf
from cubercnn.data import (
    get_omni3d_categories,
    simple_register
)
from cubercnn.util import get_cuboid_verts_faces
from cubercnn import util

"""
This file contains
* Omni3DEvaluationHelper: a helper object to accumulate and summarize evaluation results
* Omni3DEval: a wrapper around COCOeval to perform 3D bounding evaluation in the detection setting
* Omni3DEvaluator: a wrapper around COCOEvaluator to collect results on each dataset
* Omni3DParams: parameters for the evaluation API
"""

logger = logging.getLogger(__name__)

# Defines the max cross of len(dts) * len(gts)
# which we will attempt to compute on a GPU. 
# Fallback is safer computation on a CPU. 
# 0 is disabled on GPU. 
MAX_DTS_CROSS_GTS_FOR_IOU3D = 0


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that plane vertices are coplanar.
    Returns a bool tensor of size B, where True indicates a box is coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
    """
    Checks that the sides of the box have a non zero area.
    Returns a bool tensor of size B, where True indicates a box is nonzero.
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return (face_areas > eps).all(1).view(B)

def box3d_overlap(
    boxes_dt: torch.Tensor, boxes_gt: torch.Tensor, 
    eps_coplanar: float = 1e-4, eps_nonzero: float = 1e-8
) -> torch.Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero 
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero  = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print('Warning: skipping {:d} non-coplanar boxes at eval.'.format(int(invalid_coplanar.float().sum())))
    
    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print('Warning: skipping {:d} zero volume boxes at eval.'.format(int(invalid_nonzero.float().sum())))

    return ious

def calculate_metrics(results_cat, categories):
    metrics = {
        'AP2D': 0.0,
        'AP3D': 0.0, 
        'AR2D': 0.0,
        'AR3D': 0.0
    }
    
    for cat in categories:
        cat_results = results_cat[cat]
        metrics['AP2D'] += cat_results['AP2D']
        metrics['AP3D'] += cat_results['AP3D']
        metrics['AR2D'] += cat_results['AR2D'] 
        metrics['AR3D'] += cat_results['AR3D']
    
    # Calculate averages
    num_cats = len(categories)
    for k in metrics:
        metrics[k] /= num_cats
        
    return metrics

class Omni3DEvaluationHelper:
    def __init__(self, 
            dataset_names, 
            filter_settings, 
            output_folder,
            iter_label='-',
            only_2d=False,
            eval_categories=None
        ):
        """
        A helper class to initialize, evaluate and summarize Omni3D metrics. 

        The evaluator relies on the detectron2 MetadataCatalog for keeping track 
        of category names and contiguous IDs. Hence, it is important to set 
        these variables appropriately. 
        
        # (list[str]) the category names in their contiguous order
        MetadataCatalog.get('omni3d_model').thing_classes = ... 

        # (dict[int: int]) the mapping from Omni3D category IDs to the contiguous order
        MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id

        Args:
            dataset_names (list[str]): the individual dataset splits for evaluation
            filter_settings (dict): the filter settings used for evaluation, see
                cubercnn/data/datasets.py get_filter_settings_from_cfg
            output_folder (str): the output folder where results can be stored to disk.
            iter_label (str): an optional iteration/label used within the summary
            only_2d (bool): whether the evaluation mode should be 2D or 2D and 3D.
        """
        self._logger = logging.getLogger(__name__)
        self.dataset_names = dataset_names
        self.filter_settings = filter_settings
        self.output_folder = output_folder
        self.iter_label = iter_label
        self.only_2d = only_2d

        # Each dataset evaluator is stored here
        self.evaluators = OrderedDict()

        # These are the main evaluation results
        self.results = OrderedDict()

        # These store store per-dataset results to be printed
        self.results_analysis = OrderedDict()
        self.results_omni3d = OrderedDict()

        self.overall_imgIds = set()
        self.overall_catIds = set()
        # These store the evaluations for each category and area,
        # concatenated from ALL evaluated datasets. Doing so avoids
        # the need to re-compute them when accumulating results.
        self.evals_per_cat_area2D = {}
        self.evals_per_cat_area3D = {}
        self.evals_nhd_accumulators3D = {}
        self.output_folders = {
            dataset_name: os.path.join(self.output_folder, dataset_name)
            for dataset_name in dataset_names
        }

        for dataset_name in self.dataset_names:
            filter_settings['category_names'] = list(eval_categories)
            # register any datasets that need it
            if MetadataCatalog.get(dataset_name).get('json_file') is None:
                simple_register(dataset_name, filter_settings, filter_empty=False)
            
            # create an individual dataset evaluator
            self.evaluators[dataset_name] = Omni3DEvaluator(
                dataset_name, output_dir=self.output_folders[dataset_name], 
                filter_settings=filter_settings, only_2d=self.only_2d, 
                eval_prox=('Objectron' in dataset_name or 'SUNRGBD' in dataset_name),
                distributed=False, # actual evaluation should be single process
            )

            self.evaluators[dataset_name].reset()
            self.overall_imgIds.update(set(self.evaluators[dataset_name]._omni_api.getImgIds()))
            self.overall_catIds.update(set(self.evaluators[dataset_name]._omni_api.getCatIds()))
        
    def add_predictions(self, dataset_name, predictions):
        """
        Adds predictions to the evaluator for dataset_name. This can be any number of
        predictions, including all predictions passed in at once or in batches. 

        Args:
            dataset_name (str): the dataset split name which the predictions belong to
            predictions (list[dict]): each item in the list is a dict as follows:

                {
                    "image_id": <int> the unique image identifier from Omni3D,
                    "K": <np.array> 3x3 intrinsics matrix for the image,
                    "width": <int> image width,
                    "height": <int> image height,
                    "instances": [
                        {
                            "image_id":  <int> the unique image identifier from Omni3D,
                            "category_id": <int> the contiguous category prediction IDs, 
                                which can be mapped from Omni3D's category ID's using
                                MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
                            "bbox": [float] 2D box as [x1, y1, x2, y2] used for IoU2D,
                            "score": <float> the confidence score for the object,
                            "depth": <float> the depth of the center of the object,
                            "bbox3D": list[list[float]] 8x3 corner vertices used for IoU3D,
                        }
                        ...
                    ]
                }
        """
        # concatenate incoming predictions
        self.evaluators[dataset_name]._predictions += predictions

    def save_predictions(self, dataset_name):
        """
        Saves the predictions from dataset_name to disk, in a self.output_folder.

        Args:
            dataset_name (str): the dataset split name which should be saved.
        """
        # save predictions to disk
        output_folder_dataset = self.output_folders[dataset_name]
        PathManager.mkdirs(output_folder_dataset)
        file_path = os.path.join(output_folder_dataset, "instances_predictions.pth")
        with PathManager.open(file_path, "wb") as f:
            torch.save(self.evaluators[dataset_name]._predictions, f)

    def evaluate(self, dataset_name):
        """
        Runs the evaluation for an individual dataset split, assuming all 
        predictions have been passed in. 

        Args:
            dataset_name (str): the dataset split name which should be evalated.
        """
        
        if not dataset_name in self.results:
            # run evaluation and cache
            self.results[dataset_name] = self.evaluators[dataset_name].evaluate()

        results = self.results[dataset_name]
        
        # 添加调试信息
        self._logger.info(f"\nEvaluation results for {dataset_name}:")
        self._logger.info(f"Available keys in results: {list(results.keys())}")
        
        if 'bbox_2D_evals_per_cat_area' not in results:
            self._logger.error("Missing key 'bbox_2D_evals_per_cat_area' in results")
            self._logger.info("Results content:")
            for key, value in results.items():
                self._logger.info(f"  {key}: {type(value)}")
                if isinstance(value, dict):
                    self._logger.info(f"    Dict keys: {list(value.keys())}")
            return False

        logger.info('\n'+results['log_str_2D'].replace('mode=2D', '{} iter={} mode=2D'.format(dataset_name, self.iter_label)))
            
        # store the partially accumulated evaluations per category per area
        for key, item in results['bbox_2D_evals_per_cat_area'].items():
            if not key in self.evals_per_cat_area2D:
                self.evals_per_cat_area2D[key] = []
            self.evals_per_cat_area2D[key] += item

        if not self.only_2d:
            # store the partially accumulated evaluations per category per area
            for key, item in results['bbox_3D_evals_per_cat_area'].items():
                if not key in self.evals_per_cat_area3D:
                    self.evals_per_cat_area3D[key] = []
                self.evals_per_cat_area3D[key] += item
            for key, item in results["bbox_3D_nhd_accumulators"].items():
                if not key in self.evals_nhd_accumulators3D:
                    self.evals_nhd_accumulators3D[key] = []
                self.evals_nhd_accumulators3D[key] += item
            logger.info('\n'+results['log_str_3D'].replace('mode=3D', '{} iter={} mode=3D'.format(dataset_name, self.iter_label)))

        # full model category names
        category_names = self.filter_settings['category_names']

        # The set of categories present in the dataset; there should be no duplicates 
        categories = {cat for cat in category_names if 'AP-{}'.format(cat) in results['bbox_2D']}
        assert len(categories) == len(set(categories)) 

        # default are all NaN
        general_2D_AP, general_2D_AR, general_3D_AP, general_3D_AR, omni_2D_AP, omni_2D_AR, omni_3D_AP, omni_3D_AR = (np.nan,) * 8

        # 2D and 3D performance for categories in dataset; and log
        general_2D_AP = np.mean([results['bbox_2D']['AP-{}'.format(cat)] for cat in categories])
        general_2D_AR = np.mean([results['bbox_2D']['AR-{}'.format(cat)] for cat in categories])
        if not self.only_2d:
            general_3D_AP = np.mean([results['bbox_3D']['AP-{}'.format(cat)] for cat in categories])
            general_3D_AR = np.mean([results['bbox_3D']['AR-{}'.format(cat)] for cat in categories])

        # 2D and 3D performance on Omni3D categories
        omni3d_dataset_categories = get_omni3d_categories(dataset_name)  # dataset-specific categories
        if len(omni3d_dataset_categories - categories) == 0:  # omni3d_dataset_categories is a subset of categories
            omni_2D_AP = np.mean([results['bbox_2D']['AP-{}'.format(cat)] for cat in omni3d_dataset_categories])
            omni_2D_AR = np.mean([results['bbox_2D']['AR-{}'.format(cat)] for cat in omni3d_dataset_categories])
            if not self.only_2d:
                omni_3D_AP = np.mean([results['bbox_3D']['AP-{}'.format(cat)] for cat in omni3d_dataset_categories])
                omni_3D_AR = np.mean([results['bbox_3D']['AR-{}'.format(cat)] for cat in omni3d_dataset_categories])
        
        self.results_omni3d[dataset_name] = {"iters": self.iter_label, "AP2D": omni_2D_AP, "AR2D": omni_2D_AR, "AP3D": omni_3D_AP, "AR3D": omni_3D_AR}

        # Performance analysis
        extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,)*6
        if not self.only_2d:
            extras_AP15 = results['bbox_3D']['AP15']
            extras_AP25 = results['bbox_3D']['AP25']
            extras_AP50 = results['bbox_3D']['AP50']
            extras_APn = results['bbox_3D']['APn']
            extras_APm = results['bbox_3D']['APm']
            extras_APf = results['bbox_3D']['APf']

        self.results_analysis[dataset_name] = {
            "iters": self.iter_label, 
            "AP2D": general_2D_AP, "AP3D": general_3D_AP, 
            "AP3D@15": extras_AP15, "AP3D@25": extras_AP25, "AP3D@50": extras_AP50, 
            "AP3D-N": extras_APn, "AP3D-M": extras_APm, "AP3D-F": extras_APf,
            "AR2D": general_2D_AR, "AR3D": general_3D_AR
        }

        # Performance per category
        results_cat = OrderedDict()
        for cat in category_names:
            cat_2D_AP, cat_2D_AR, cat_3D_AP, cat_3D_AR = (np.nan,) * 4
            if 'AP-{}'.format(cat) in results['bbox_2D']:
                cat_2D_AP = results['bbox_2D']['AP-{}'.format(cat)]
                if not self.only_2d:
                    cat_3D_AP = results['bbox_3D']['AP-{}'.format(cat)]
            if 'AR-{}'.format(cat) in results['bbox_2D']:
                cat_2D_AR = results['bbox_2D']['AR-{}'.format(cat)]
                if not self.only_2d:
                    cat_3D_AR = results['bbox_3D']['AR-{}'.format(cat)]
            if not np.isnan(cat_2D_AP) or not np.isnan(cat_3D_AP) or not np.isnan(cat_2D_AR) or not np.isnan(cat_3D_AR):
                results_cat[cat] = {"AP2D": cat_2D_AP, "AP3D": cat_3D_AP, "AR2D": cat_2D_AR, "AR3D": cat_3D_AR}
        utils_logperf.print_ap_category_histogram(dataset_name, results_cat)

    def summarize_all(self,):
        '''
        Report collective metrics when possible for the the Omni3D dataset.
        This uses pre-computed evaluation results from each dataset, 
        which were aggregated and cached while evaluating individually. 
        This process simply re-accumulate and summarizes them. 
        '''

        # First, double check that we have all the evaluations
        for dataset_name in self.dataset_names:
            if not dataset_name in self.results:
                self.evaluate(dataset_name)

        if self.dataset_names[0].endswith(("_novel", "_test")):
            category_path = "configs/category_meta.json" # TODO: hard coded
            metadata = util.load_json(category_path)
            thing_classes = metadata['thing_classes']
            catId2contiguous = {int(key):val for key, val in metadata['thing_dataset_id_to_contiguous_id'].items()}
        else:
            thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
            catId2contiguous = MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id
        ordered_things = [thing_classes[catId2contiguous[cid]] for cid in self.overall_catIds]
        categories = set(ordered_things)

        evaluator2D = Omni3Deval(mode='2D')
        evaluator2D.params.catIds = list(self.overall_catIds)
        evaluator2D.params.imgIds = list(self.overall_imgIds)
        evaluator2D.evalImgs = True
        evaluator2D.evals_per_cat_area = self.evals_per_cat_area2D
        evaluator2D._paramsEval = copy.deepcopy(evaluator2D.params)
        evaluator2D.accumulate()
        summarize_str2D = evaluator2D.summarize()
        
        precisions = evaluator2D.eval['precision']
        recalls = evaluator2D.eval['recall']

        metrics = ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl", "AR1", "AR10", "AR100",]

        results2D = {
            metric: float(
                evaluator2D.stats[idx] * 100 if evaluator2D.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }

        for idx, name in enumerate(ordered_things):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]

            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]

            ap = np.mean(precision) if precision.size else float("nan")
            ar = np.mean(recall) if recall.size else float("nan")
            results2D.update({"AP-" + "{}".format(name): float(ap * 100)})
            results2D.update({"AR-" + "{}".format(name): float(ar * 100)})

        evaluator3D = Omni3Deval(mode='3D')
        evaluator3D.params.catIds = list(self.overall_catIds)
        evaluator3D.params.imgIds = list(self.overall_imgIds)
        evaluator3D.evalImgs = True
        evaluator3D.evals_per_cat_area = self.evals_per_cat_area3D
        evaluator3D._paramsEval = copy.deepcopy(evaluator3D.params)
        evaluator3D.accumulate()
        summarize_str3D = evaluator3D.summarize()
        
        precisions = evaluator3D.eval['precision']
        recalls = evaluator3D.eval['recall']

        metrics = ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf", "AR1", "AR10", "AR100",]

        results3D = {
            metric: float(
                evaluator3D.stats[idx] * 100 if evaluator3D.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }

        for idx, name in enumerate(ordered_things):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]

            ap = np.mean(precision) if precision.size else float("nan")
            ar = np.mean(recall) if recall.size else float("nan")
            results3D.update({"AP-" + "{}".format(name): float(ap * 100)})
            results3D.update({"AR-" + "{}".format(name): float(ar * 100)})


        # All concat categories
        general_2D_AP, general_2D_AR, general_3D_AP, general_3D_AR, = (np.nan,) * 4

        general_2D_AP = np.mean([results2D['AP-{}'.format(cat)] for cat in categories])
        general_2D_AR = np.mean([results2D['AR-{}'.format(cat)] for cat in categories])
        if not self.only_2d:
            general_3D_AP = np.mean([results3D['AP-{}'.format(cat)] for cat in categories])
            general_3D_AR = np.mean([results3D['AR-{}'.format(cat)] for cat in categories])

        # Analysis performance
        extras_AP15, extras_AP25, extras_AP50, extras_APn, extras_APm, extras_APf = (np.nan,) * 6
        if not self.only_2d:
            extras_AP15 = results3D['AP15']
            extras_AP25 = results3D['AP25']
            extras_AP50 = results3D['AP50']
            extras_APn = results3D['APn']
            extras_APm = results3D['APm']
            extras_APf = results3D['APf']

        self.results_analysis["<Concat>"] = {
            "iters": self.iter_label, 
            "AP2D": general_2D_AP, "AP3D": general_3D_AP, 
            "AP3D@15": extras_AP15, "AP3D@25": extras_AP25, "AP3D@50": extras_AP50, 
            "AP3D-N": extras_APn, "AP3D-M": extras_APm, "AP3D-F": extras_APf, 
            "AR2D": general_2D_AR, "AR3D": general_3D_AR
        }
        overall_NHD = np.mean(self.evals_nhd_accumulators3D["overall"])
        disent_xy_nhd = np.mean(self.evals_nhd_accumulators3D["xy"])
        disent_z_nhd = np.mean(self.evals_nhd_accumulators3D["z"])
        disent_dims_nhd = np.mean(self.evals_nhd_accumulators3D["dimensions"])
        disent_pose_nhd = np.mean(self.evals_nhd_accumulators3D["pose"])

        # Omni3D Outdoor performance
        omni_2D, omni_3D, omni_2D_AR, omni_3D_AR = (np.nan,) * 4

        omni3d_outdoor_categories = get_omni3d_categories("omni3d_out")
        if len(omni3d_outdoor_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_outdoor_categories])
            omni_2D_AR = np.mean([results2D['AR-{}'.format(cat)] for cat in omni3d_outdoor_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_outdoor_categories])
                omni_3D_AR = np.mean([results3D['AR-{}'.format(cat)] for cat in omni3d_outdoor_categories])

        self.results_omni3d["Omni3D_Out"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D, "AR2D": omni_2D_AR, "AR3D": omni_3D_AR}

        # Omni3D Indoor performance
        omni_2D, omni_3D, omni_2D_AR, omni_3D_AR = (np.nan,) * 4

        omni3d_indoor_categories = get_omni3d_categories("omni3d_in")
        if len(omni3d_indoor_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_indoor_categories])
            omni_2D_AR = np.mean([results2D['AR-{}'.format(cat)] for cat in omni3d_indoor_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_indoor_categories])
                omni_3D_AR = np.mean([results3D['AR-{}'.format(cat)] for cat in omni3d_indoor_categories])

        self.results_omni3d["Omni3D_In"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D, "AR2D": omni_2D_AR, "AR3D": omni_3D_AR}

        # Omni3D performance
        omni_2D, omni_3D, omni_2D_AR, omni_3D_AR = (np.nan,) * 4

        omni3d_categories = get_omni3d_categories("omni3d")
        if len(omni3d_categories - categories) == 0:
            omni_2D = np.mean([results2D['AP-{}'.format(cat)] for cat in omni3d_categories])
            omni_2D_AR = np.mean([results2D['AR-{}'.format(cat)] for cat in omni3d_indoor_categories])
            if not self.only_2d:
                omni_3D = np.mean([results3D['AP-{}'.format(cat)] for cat in omni3d_categories])
                omni_3D_AR = np.mean([results3D['AR-{}'.format(cat)] for cat in omni3d_indoor_categories])

        self.results_omni3d["Omni3D"] = {"iters": self.iter_label, "AP2D": omni_2D, "AP3D": omni_3D, "AR2D": omni_2D_AR, "AR3D": omni_3D_AR}

        # Per-category performance for the cumulative datasets
        results_cat = OrderedDict()
        for cat in self.filter_settings['category_names']:
            cat_2D, cat_3D, cat_2D_AR, cat_3D_AR, = (np.nan,) * 4
            if 'AP-{}'.format(cat) in results2D:
                cat_2D = results2D['AP-{}'.format(cat)]
                cat_2D_AR = results2D['AR-{}'.format(cat)]
                if not self.only_2d:
                    cat_3D = results3D['AP-{}'.format(cat)]
                    cat_3D_AR = results3D['AR-{}'.format(cat)]
            if not np.isnan(cat_2D) or not np.isnan(cat_3D) or not np.isnan(cat_2D_AR) or not np.isnan(cat_3D_AR):
                results_cat[cat] = {"AP2D": cat_2D, "AP3D": cat_3D, "AR2D": cat_2D_AR, "AR3D": cat_3D_AR}
        utils_logperf.print_ap_category_histogram("<Concat>", results_cat)
        # This is only for the novel categories
        if set(results_cat.keys()) == set(['monitor', 'bag', 'dresser', 'board', 'printer', 'keyboard', 'painting', 'drawers', 'microwave', 'computer', 'kitchen pan', 'potted plant', 'tissues', 'rack', 'tray', 'toys', 'phone', 'podium', 'cart', 'soundsystem', 'fireplace', 'tram']):
            easy_novel_categories = set(['board', 'printer', 'painting', 'microwave', 'tray', 'podium', 'cart', 'tram'])
            hard_novel_categories = set(results_cat.keys()) - easy_novel_categories
            easy_metrics = calculate_metrics(results_cat, easy_novel_categories)
            hard_metrics = calculate_metrics(results_cat, hard_novel_categories)
            logger.info(f"Easy Novel Categories: {easy_novel_categories}") 
            logger.info(f"Hard Novel Categories: {hard_novel_categories}")
            easy_metrics_formatted = {k: f"{v:.2f}" for k, v in easy_metrics.items()}
            hard_metrics_formatted = {k: f"{v:.2f}" for k, v in hard_metrics.items()}
            # logger.info(f"Easy Novel Categories Metrics: {easy_metrics_formatted}")
            # logger.info(f"Hard Novel Categories Metrics: {hard_metrics_formatted}")
            utils_logperf.print_ap_hard_easy_for_novel(easy_metrics_formatted, hard_metrics_formatted)
        utils_logperf.print_ap_analysis_histogram(self.results_analysis)
        utils_logperf.print_ap_omni_histogram(self.results_omni3d)
        logger.info(f"Overall NHD: {overall_NHD}")
        logger.info(f"Disentangled XY NHD: {disent_xy_nhd}")
        logger.info(f"Disentangled  Z NHD: {disent_z_nhd}")
        logger.info(f"Disentangled Dimensions NHD: {disent_dims_nhd}")
        logger.info(f"Disentangled Pose NHD: {disent_pose_nhd}")
    def add_cat_name_to_predictions(self, predictions, category_names_official):
        for img in predictions:
            for instance in img["instances"]:
                instance["category_name"] = category_names_official[instance["category_id"]]

def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader. 
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.
    """
    
    num_devices = get_world_size()
    distributed = num_devices > 1
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0

    inference_json = []

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            if "depth" in inputs[0]:
                depth = torch.stack([x["depth"] for x in inputs])
                if torch.cuda.is_available():
                    depth = depth.cuda()
                outputs = model(inputs, prompt_depth=depth)
            else:
                outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()

            for input, output in zip(inputs, outputs):
                prediction = {
                    "image_id": input.get("image_id", input.get("file_name", str(idx))),
                    "K": input["K"],
                    "width": input["width"],
                    "height": input["height"],
                }

                instances = output["instances"].to('cpu')
                prediction["instances"] = evaluator.instances_to_coco_json(instances, prediction["image_id"])
                inference_json.append(prediction)

            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # 计算总时间
    total_time = time.perf_counter() - start_time

    # 只保留最基本的统计信息
    logger.info(f"\nFinal Statistics:")
    logger.info(f"Total inference time: {str(datetime.timedelta(seconds=int(total_time)))} ({total_time / (total - num_warmup):.6f} s / iter per device, on {num_devices} devices)")
    logger.info(f"Total inference pure compute time: {str(datetime.timedelta(seconds=int(total_compute_time)))} ({total_compute_time / (total - num_warmup):.6f} s / iter per device, on {num_devices} devices)")

    if distributed:
        comm.synchronize()
        inference_json = comm.gather(inference_json, dst=0)
        inference_json = list(itertools.chain(*inference_json))

        if not comm.is_main_process():
            return []

    # 添加调试信息
    logger.info(f"Number of predictions collected: {len(inference_json)}")
    if inference_json:
        logger.info(f"Sample prediction structure:")
        logger.info(f"Keys in first prediction: {list(inference_json[0].keys())}")
        logger.info(f"Number of instances in first prediction: {len(inference_json[0]['instances'])}")
        if inference_json[0]['instances']:
            logger.info(f"Keys in first instance: {list(inference_json[0]['instances'][0].keys())}")

    return inference_json

class Omni3DEvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=False,
        eval_prox=False,
        only_2d=False,
        filter_settings={},
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. For now, support only for "bbox".
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                    contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            eval_prox (bool): whether to perform proximity evaluation. For datasets that are not
                exhaustively annotated.
            only_2d (bool): evaluates only 2D performance if set to True
            filter_settions: settings for the dataset loader. TBD
        """

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl
        self._eval_prox = eval_prox
        self._only_2d = only_2d
        self._filter_settings = filter_settings

        # COCOeval requires the limit on the number of detections per image (maxDets) to be a list
        # with at least 3 elements. The default maxDets in COCOeval is [1, 10, 100], in which the
        # 3rd element (100) is used as the limit on the number of detections per image when
        # evaluating AP. COCOEvaluator expects an integer for max_dets_per_image, so for COCOeval,
        # we reformat max_dets_per_image into [1, 10, max_dets_per_image], based on the defaults.
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]

        else:
            max_dets_per_image = [1, 10, max_dets_per_image]

        self._max_dets_per_image = max_dets_per_image

        self._tasks = tasks
        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._omni_api = Omni3D([json_file], filter_settings)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._omni_api.dataset

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """

        # Optional image keys to keep when available
        img_keys_optional = ["p2"]

        for input, output in zip(inputs, outputs):

            prediction = {
                "image_id": input["image_id"],
                "K": input["K"],
                "width": input["width"],
                "height": input["height"],
            }

            # store optional keys when available
            for img_key in img_keys_optional:
                if img_key in input:
                    prediction.update({img_key: input[img_key]})

            # already in COCO format
            if type(output["instances"]) == list:
                prediction["instances"] = output["instances"]

            # tensor instances format
            else: 
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = self.instances_to_coco_json(
                    instances, input["image_id"]
                )

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _derive_omni_results(self, omni_eval, iou_type, mode, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            omni_eval (None or Omni3Deval): None represents no predictions from model.
            iou_type (str):
            mode (str): either "2D" or "3D"
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """
        assert mode in ["2D", "3D"]

        metrics = {
            "2D": ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl", "AR1", "AR10", "AR100",],
            "3D": ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf", "AR1", "AR10", "AR100",],
        }[mode]

        if iou_type != "bbox":
            raise ValueError("Support only for bbox evaluation.")

        if omni_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                omni_eval.stats[idx] * 100 if omni_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        if mode == "3D":
            addtional_metrics = ["overall", "xy", "z", "dimensions", "pose"]
            for metric in addtional_metrics:
                if metric == "overall":
                    results[metric + "_NHD"] = float(omni_eval.eval["average_nhd"][metric] if omni_eval.eval["average_nhd"][metric] >= 0 else "nan")
                else:
                    results["disent_"+metric + "_NHD"] = float(omni_eval.eval["average_nhd"][metric] if omni_eval.eval["average_nhd"][metric] >= 0 else "nan")
        self._logger.info(
            "Evaluation results for {} in {} mode: \n".format(iou_type, mode)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP and AR
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = omni_eval.eval["precision"]
        recalls = omni_eval.eval["recall"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")

            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")

            results_per_category.append(("{} (AP)".format(name), float(ap * 100)))
            results_per_category.append(("{} (AR)".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(4, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_table = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_table,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "metric"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(
            "Per-category {} AP/AR in {} mode: \n".format(iou_type, mode) + table
        )
        results.update({"AP-" + name.replace(" (AP)", ""): ap for name, ap in results_per_category if "(AP)" in name})
        results.update({"AR-" + name.replace(" (AR)", ""): ar for name, ar in results_per_category if "(AR)" in name})

        return results

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._logger.info(f"Number of predictions to process: {len(predictions)}")
        
        # 添加更详细的预测信息日志
        if predictions:
            self._logger.info("First prediction structure:")
            self._logger.info(f"Keys: {list(predictions[0].keys())}")
            if 'instances' in predictions[0]:
                self._logger.info(f"Number of instances: {len(predictions[0]['instances'])}")
                if predictions[0]['instances']:
                    self._logger.info(f"Instance keys: {list(predictions[0]['instances'][0].keys())}")
                    self._logger.info(f"Category ID of first instance: {predictions[0]['instances'][0]['category_id']}")

        # 创建文件名到image_id的映射
        self.filename_to_id = {
            img['file_path'].split('/')[-1]: img['id']
            for img in self._omni_api.dataset['images']
        }
        self._logger.info(f"Created filename to ID mapping for {len(self.filename_to_id)} images")
        
        # 创建反向映射用于保存原始映射关系
        id_to_filename = {v: k for k, v in self.filename_to_id.items()}
        
        # 保存原始image_id映射关系到数据集中
        for img in self._omni_api.dataset['images']:
            img['original_image_id'] = img['id']
        
        # 保存原始image_id映射关系到预测结果中
        self._logger.info("Processing predictions and updating image IDs...")
        processed_count = 0
        for pred in predictions:
            pred['original_image_id'] = pred['image_id']
            
            # 修改预测结果中的image_id
            if isinstance(pred['image_id'], str):
                file_name = pred['image_id'].split('/')[-1]  # 只取文件名部分
                if file_name in self.filename_to_id:
                    new_id = self.filename_to_id[file_name]
                    pred['image_id'] = new_id
                    
                    # 确保instances中的image_id与pred的image_id一致，同时保存原始id
                    for instance in pred['instances']:
                        instance['original_image_id'] = instance['image_id']
                        instance['image_id'] = new_id
            else:
                # 如果image_id已经是整数，确保instances使用相同的id
                for instance in pred['instances']:
                    instance['original_image_id'] = instance['image_id']
                    instance['image_id'] = pred['image_id']
        
            processed_count += 1
            if processed_count % 1000 == 0:
                self._logger.info(f"Processed {processed_count}/{len(predictions)} predictions")
        
        self._logger.info(f"Finished processing all {processed_count} predictions")
        
        omni_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        self._logger.info(f"Total number of instances after flattening: {len(omni_results)}")
        
        tasks = self._tasks or self._tasks_from_predictions(omni_results)
        self._logger.info(f"Tasks to evaluate: {tasks}")

        if self._metadata.name.endswith(("_novel", "_test")):
            category_path = "configs/category_meta.json" # TODO: hard coded
            self._logger.info(f"Loading category metadata from: {category_path}")
            metadata = util.load_json(category_path)
            omni3d_global_categories = metadata['thing_classes']
        else:
            omni3d_global_categories = MetadataCatalog.get('omni3d_model').thing_classes
        
        self._logger.info(f"Number of global categories: {len(omni3d_global_categories)}")

        # 过滤结果以只包含数据集中存在的类别
        dataset_results = []
        
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            self._logger.info(f"Number of classes in dataset: {num_classes}")

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            filtered_count = 0
            invalid_categories = set()
            
            # Before filtering, add debug information about the category mappings
            self._logger.info(f"Dataset thing classes: {self._metadata.thing_classes}")
            self._logger.info(f"Dataset ID to contiguous ID mapping: {dataset_id_to_contiguous_id}")
            self._logger.info(f"First few predictions category IDs: {[r['category_id'] for r in omni_results[:5]]}")

            # When filtering results, ensure proper category mapping
            for result in omni_results:
                category_id = result["category_id"]
                
                # Check if the category_id is already a dataset ID (not a contiguous ID)
                if category_id in dataset_id_to_contiguous_id:
                    # It's already a dataset ID, keep it as is
                    dataset_results.append(result)
                    filtered_count += 1
                    continue
                    
                # If it's a contiguous ID, try to map it
                if category_id < num_classes:
                    # Map contiguous ID to dataset ID
                    if category_id in reverse_id_mapping:
                        result["category_id"] = reverse_id_mapping[category_id]
                        dataset_results.append(result)
                        filtered_count += 1
                    else:
                        # Try to find the category by name
                        try:
                            cat_name = omni3d_global_categories[category_id]
                            if cat_name in self._metadata.thing_classes:
                                cat_idx = self._metadata.thing_classes.index(cat_name)
                                # Find the dataset ID for this category
                                for dataset_id, contiguous_id in dataset_id_to_contiguous_id.items():
                                    if contiguous_id == cat_idx:
                                        result["category_id"] = dataset_id
                                        dataset_results.append(result)
                                        filtered_count += 1
                                        break
                        except (IndexError, ValueError):
                            invalid_categories.add(category_id)
                else:
                    invalid_categories.add(category_id)
            
            if invalid_categories:
                self._logger.warning(f"Found predictions with invalid category IDs: {invalid_categories}")
            
            self._logger.info(f"Filtered results: {filtered_count} valid instances kept")

        # replace the results with the filtered instances
        omni_results = dataset_results

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "omni_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            
            self._logger.info(f"Final check before saving:")
            self._logger.info(f"Number of results to save: {len(omni_results)}")
            self._logger.info(f"Results type: {type(omni_results)}")
            
            if omni_results:
                self._logger.info(f"First result structure:")
                for key, value in omni_results[0].items():
                    self._logger.info(f"  {key}: {type(value)}")
                    if isinstance(value, (list, dict)):
                        self._logger.info(f"    Length/Size: {len(value)}")
            
            try:
                with PathManager.open(file_path, "w") as f:
                    json.dump(omni_results, f)
                    f.flush()
                    os.fsync(f.fileno())
                
                # 验证文件是否正确写入
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    self._logger.info(f"File successfully written, size: {file_size} bytes")
                    
                    # 读取文件验证内容
                    with PathManager.open(file_path, "r") as f:
                        saved_results = json.load(f)
                        self._logger.info(f"Successfully loaded saved results, count: {len(saved_results)}")
                        if len(saved_results) != len(omni_results):
                            self._logger.warning(f"Warning: Number of saved results ({len(saved_results)}) "
                                               f"differs from original ({len(omni_results)})")
                else:
                    self._logger.error("File was not created!")
            except Exception as e:
                self._logger.error(f"Error saving results: {str(e)}")
                import traceback
                self._logger.error(traceback.format_exc())

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox"}, f"Got unknown task: {task}!"
            
            if len(omni_results) == 0:
                self._logger.warning("No predictions found, skipping evaluation")
                # 当没有预测结果时，返回空的评估结果
                evals = {"2D": None, "3D": None}
                log_strs = {"2D": "", "3D": ""}
            else:
                # 检查GT数据集和预测结果的匹配情况
                gt_img_ids = set([img['id'] for img in self._omni_api.dataset.get('images', [])])
                pred_img_ids = set([r['image_id'] for r in omni_results])
                common_img_ids = gt_img_ids.intersection(pred_img_ids)
                
                self._logger.info(f"GT dataset has {len(gt_img_ids)} images")
                self._logger.info(f"Predictions cover {len(pred_img_ids)} images")
                self._logger.info(f"Common images between GT and predictions: {len(common_img_ids)}")
                
                gt_cat_ids = set([cat['id'] for cat in self._omni_api.dataset.get('categories', [])])
                pred_cat_ids = set([r['category_id'] for r in omni_results])
                common_cat_ids = gt_cat_ids.intersection(pred_cat_ids)
                
                self._logger.info(f"GT dataset has {len(gt_cat_ids)} categories")
                self._logger.info(f"Predictions cover {len(pred_cat_ids)} categories")
                self._logger.info(f"Common categories between GT and predictions: {len(common_cat_ids)}")
                
                if len(common_img_ids) == 0 or len(common_cat_ids) == 0:
                    self._logger.error("No common images or categories between GT and predictions!")
                    self._logger.info(f"GT image IDs (sample): {list(gt_img_ids)[:5]}")
                    self._logger.info(f"Pred image IDs (sample): {list(pred_img_ids)[:5]}")
                    self._logger.info(f"GT category IDs: {gt_cat_ids}")
                    self._logger.info(f"Pred category IDs: {pred_cat_ids}")
                
                evals, log_strs = _evaluate_predictions_on_omni(
                    self._omni_api,
                    omni_results,
                    task,
                    img_ids=img_ids,
                    only_2d=self._only_2d,
                    eval_prox=self._eval_prox,
                )

            modes = evals.keys() if evals else []
            for mode in modes:
                if evals[mode] is None:
                    continue
                res = self._derive_omni_results(
                    evals[mode],
                    task,
                    mode,
                    class_names=self._metadata.get("thing_classes"),
                )
                self._results[task + "_" + format(mode)] = res
                self._results[task + "_" + format(mode) + '_evalImgs'] = evals[mode].evalImgs
                self._results[task + "_" + format(mode) + '_evals_per_cat_area'] = evals[mode].evals_per_cat_area
                if mode == "3D":
                    self._results[task + "_" + format(mode) + '_nhd_accumulators'] = evals[mode].eval["nhd_accumulators"]
            
            if "2D" in log_strs:
                self._results["log_str_2D"] = log_strs["2D"]
            if "3D" in log_strs:
                self._results["log_str_3D"] = log_strs["3D"]

    def instances_to_coco_json(self, instances, img_id):
        """
        Convert detection results to COCO json format
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []
        
        # 确保img_id是整数
        if isinstance(img_id, str):
            # 如果是文件路径,只取文件名部分
            img_id = img_id.split('/')[-1]
            # 从filename_to_id映射中获取对应的整数id
            if hasattr(self, 'filename_to_id') and img_id in self.filename_to_id:
                img_id = self.filename_to_id[img_id]
            else:
                # 如果找不到映射,使用hash作为备选
                img_id = hash(img_id) % (10 ** 8)

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_3d = False
        if instances.has('pred_bbox3D'):
            has_3d = True
            bbox3D = instances.pred_bbox3D.numpy()
            center_cam = instances.pred_center_cam.numpy()
            center_2D = instances.pred_center_2D.numpy()
            dimensions = instances.pred_dimensions.numpy()
            pose = instances.pred_pose.numpy()

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            if has_3d:
                result["bbox3D"] = bbox3D[k].tolist()
                result["center_cam"] = center_cam[k].tolist()
                result["center_2D"] = center_2D[k].tolist()
                result["dimensions"] = dimensions[k].tolist()
                result["pose"] = pose[k].tolist()
                result["depth"] = float(center_cam[k][2])

            results.append(result)
        return results


def _evaluate_predictions_on_omni(omni_gt, omni_results, iou_type, img_ids=None, only_2d=False, eval_prox=False):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(omni_results) > 0
    log_strs, evals = {}, {}

    logger.info("Debug: Starting evaluation")
    logger.info(f"Debug: Number of results to evaluate: {len(omni_results)}")
    logger.info(f"Debug: First result sample: {omni_results[0]}")

    # 确保所有必需的字段都存在
    required_fields = ['image_id', 'category_id', 'bbox', 'score']
    for result in omni_results:
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            logger.warning(f"Missing required fields in result: {missing_fields}")
            continue

    # 确保image_id和category_id是整数
    for result in omni_results:
        if not isinstance(result['image_id'], int):
            result['image_id'] = int(float(result['image_id']))
        if not isinstance(result['category_id'], int):
            result['category_id'] = int(float(result['category_id']))

    # 直接使用loadRes方法而不是通过临时文件
    try:
        # 检查GT数据集结构
        logger.info("Debug: GT dataset structure:")
        logger.info(f"Images: {len(omni_gt.dataset.get('images', []))}")
        logger.info(f"Categories: {len(omni_gt.dataset.get('categories', []))}")
        logger.info(f"Annotations: {len(omni_gt.dataset.get('annotations', []))}")
        
        # 确保结果中的image_id存在于GT数据集中
        gt_img_ids = set([img['id'] for img in omni_gt.dataset.get('images', [])])
        valid_results = [r for r in omni_results if r['image_id'] in gt_img_ids]
        
        if len(valid_results) < len(omni_results):
            logger.warning(f"Filtered out {len(omni_results) - len(valid_results)} results with invalid image_ids")
            omni_results = valid_results
        
        # 确保结果中的category_id存在于GT数据集中
        gt_cat_ids = set([cat['id'] for cat in omni_gt.dataset.get('categories', [])])
        valid_results = [r for r in omni_results if r['category_id'] in gt_cat_ids]
        
        if len(valid_results) < len(omni_results):
            logger.warning(f"Filtered out {len(omni_results) - len(valid_results)} results with invalid category_ids")
            omni_results = valid_results
        
        # 直接使用loadRes方法
        omni_dt = omni_gt.loadRes(omni_results)
        logger.info("Debug: Successfully loaded results into COCO format")
        
    except Exception as e:
        logger.error(f"Debug: Error loading results: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    modes = ["2D"] if only_2d else ["2D", "3D"]
    logger.info(f"Debug: Evaluating modes: {modes}")

    for mode in modes:
        logger.info(f"\nDebug: Processing mode {mode}")
        try:
            omni_eval = Omni3DevalWithNHD(
                omni_gt, omni_dt, 
                iouType=iou_type, 
                mode=mode, 
                eval_prox=eval_prox
            )
            logger.info("Debug: Created evaluator")

            if img_ids is not None:
                omni_eval.params.imgIds = img_ids

            omni_eval.evaluate()
            logger.info("Debug: Finished evaluate()")

            omni_eval.accumulate()
            logger.info("Debug: Finished accumulate()")

            log_str = omni_eval.summarize()
            logger.info("Debug: Finished summarize()")

            log_strs[mode] = log_str
            evals[mode] = omni_eval

        except Exception as e:
            logger.error(f"Debug: Error in mode {mode}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    return evals, log_strs


# ---------------------------------------------------------------------
#                               Omni3DParams
# ---------------------------------------------------------------------
class Omni3DParams:
    """
    Params for the Omni evaluation API
    """

    def setDet2DParams(self):
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]

        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setDet3DParams(self):
        self.imgIds = []
        self.catIds = []

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            0.05, 0.5, int(np.round((0.5 - 0.05) / 0.05)) + 1, endpoint=True
        )

        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

        self.maxDets = [1, 10, 100]
        self.areaRng = [[0, 1e5], [0, 10], [10, 35], [35, 1e5]]
        self.areaRngLbl = ["all", "near", "medium", "far"]
        self.useCats = 1

    def __init__(self, mode="2D"):
        """
        Args:
            iouType (str): defines 2D or 3D evaluation parameters.
                One of {"2D", "3D"}
        """

        if mode == "2D":
            self.setDet2DParams()

        elif mode == "3D":
            self.setDet3DParams()

        else:
            raise Exception("mode %s not supported" % (mode))

        self.iouType = "bbox"
        self.mode = mode
        # the proximity threshold defines the neighborhood
        # when evaluating on non-exhaustively annotated datasets
        self.proximity_thresh = 0.3


# ---------------------------------------------------------------------
#                               Omni3Deval
# ---------------------------------------------------------------------
class Omni3Deval(COCOeval):
    """
    Wraps COCOeval for 2D or 3D box evaluation depending on mode
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="bbox", mode="2D", eval_prox=False
    ):
        """
        Initialize COCOeval using coco APIs for Gt and Dt
        Args:
            cocoGt: COCO object with ground truth annotations
            cocoDt: COCO object with detection results
            iouType: (str) defines the evaluation type. Supports only "bbox" now.
            mode: (str) defines whether to evaluate 2D or 3D performance.
                One of {"2D", "3D"}
            eval_prox: (bool) if True, performs "Proximity Evaluation", i.e.
                evaluates detections in the proximity of the ground truth2D boxes.
                This is used for datasets which are not exhaustively annotated.
        """
        if not iouType:
            print("iouType not specified. use default iouType bbox")
        elif iouType != "bbox":
            print("no support for %s iouType" % (iouType))
        self.mode = mode
        if mode not in ["2D", "3D"]:
            raise Exception("mode %s not supported" % (mode))
        self.eval_prox = eval_prox
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs = defaultdict(list) 

        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Omni3DParams(mode)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.evals_per_cat_area = None

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        """
        
        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        ignore_flag = "ignore2D" if self.mode == "2D" else "ignore3D"
        for gt in gts:
            gt[ignore_flag] = gt[ignore_flag] if ignore_flag in gt else 0

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''

        print('Accumulating evaluation results...')
        assert self.evalImgs, 'Please run evaluate() first'

        tic = time.time()

        # allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]

        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)

        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval

        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        # get inds to evaluate
        catid_list = [k for n, k in enumerate(p.catIds)  if k in setK]
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]

        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        has_precomputed_evals = not (self.evals_per_cat_area is None)
        
        if has_precomputed_evals:
            evals_per_cat_area = self.evals_per_cat_area
        else:
            evals_per_cat_area = {}

        # retrieve E at each category, area range, and max number of detections
        for k, (k0, catId) in enumerate(zip(k_list, catid_list)):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0

                if has_precomputed_evals:
                    E = evals_per_cat_area[(catId, a)]

                else:
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    evals_per_cat_area[(catId, a)] = E

                if len(E) == 0:
                    continue

                for m, maxDet in enumerate(m_list):

                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0)

                    if npig == 0:
                        continue

                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]

                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass

                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)

        self.evals_per_cat_area = evals_per_cat_area

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs
        """
        self._logger.info("Starting evaluation...")
        
        if len(self._predictions) == 0:
            self._logger.warning("No predictions found, skipping evaluation")
            return {}
            
        self._logger.info(f"Evaluating {len(self._predictions)} predictions")
        
        # 添加调试信息
        if self._predictions:
            self._logger.info("Sample prediction structure:")
            sample_pred = self._predictions[0]
            for key, value in sample_pred.items():
                self._logger.info(f"  {key}: {type(value)}")
                if isinstance(value, list) and value:
                    self._logger.info(f"    First item in list: {value[0]}")

        # ... existing evaluation code ...

        # 在返回结果之前添加调试信息
        results = self._derive_coco_results()
        self._logger.info("Evaluation completed")
        self._logger.info(f"Results keys: {list(results.keys())}")
        
        # 确保必要的键存在
        if 'bbox_2D_evals_per_cat_area' not in results:
            self._logger.error("Required key 'bbox_2D_evals_per_cat_area' not found in results")
            self._logger.info("Available evaluation data:")
            if hasattr(self, 'eval'):
                self._logger.info(f"  eval keys: {list(self.eval.keys())}")
            if hasattr(self, '_coco_eval'):
                self._logger.info(f"  _coco_eval attributes: {dir(self._coco_eval)}")
        
        return results

    def _derive_omni_results(self, omni_eval, iou_type, mode, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            omni_eval (None or Omni3Deval): None represents no predictions from model.
            iou_type (str):
            mode (str): either "2D" or "3D"
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """
        assert mode in ["2D", "3D"]

        metrics = {
            "2D": ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl", "AR1", "AR10", "AR100",],
            "3D": ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf", "AR1", "AR10", "AR100",],
        }[mode]

        if iou_type != "bbox":
            raise ValueError("Support only for bbox evaluation.")

        if omni_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                omni_eval.stats[idx] * 100 if omni_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        if mode == "3D":
            addtional_metrics = ["overall", "xy", "z", "dimensions", "pose"]
            for metric in addtional_metrics:
                if metric == "overall":
                    results[metric + "_NHD"] = float(omni_eval.eval["average_nhd"][metric] if omni_eval.eval["average_nhd"][metric] >= 0 else "nan")
                else:
                    results["disent_"+metric + "_NHD"] = float(omni_eval.eval["average_nhd"][metric] if omni_eval.eval["average_nhd"][metric] >= 0 else "nan")
        self._logger.info(
            "Evaluation results for {} in {} mode: \n".format(iou_type, mode)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP and AR
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = omni_eval.eval["precision"]
        recalls = omni_eval.eval["recall"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")

            recall = recalls[:, idx, 0, -1]
            recall = recall[recall > -1]
            ar = np.mean(recall) if recall.size else float("nan")

            results_per_category.append(("{} (AP)".format(name), float(ap * 100)))
            results_per_category.append(("{} (AR)".format(name), float(ar * 100)))

        # tabulate it
        N_COLS = min(4, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_table = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_table,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "metric"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info(
            "Per-category {} AP/AR in {} mode: \n".format(iou_type, mode) + table
        )
        results.update({"AP-" + name.replace(" (AP)", ""): ap for name, ap in results_per_category if "(AP)" in name})
        results.update({"AR-" + name.replace(" (AR)", ""): ar for name, ar in results_per_category if "(AR)" in name})

        return results

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._logger.info(f"Number of predictions to process: {len(predictions)}")
        
        # 添加更详细的预测信息日志
        if predictions:
            self._logger.info("First prediction structure:")
            self._logger.info(f"Keys: {list(predictions[0].keys())}")
            if 'instances' in predictions[0]:
                self._logger.info(f"Number of instances: {len(predictions[0]['instances'])}")
                if predictions[0]['instances']:
                    self._logger.info(f"Instance keys: {list(predictions[0]['instances'][0].keys())}")
                    self._logger.info(f"Category ID of first instance: {predictions[0]['instances'][0]['category_id']}")

        # 创建文件名到image_id的映射
        self.filename_to_id = {
            img['file_path'].split('/')[-1]: img['id']
            for img in self._omni_api.dataset['images']
        }
        self._logger.info(f"Created filename to ID mapping for {len(self.filename_to_id)} images")
        
        # 创建反向映射用于保存原始映射关系
        id_to_filename = {v: k for k, v in self.filename_to_id.items()}
        
        # 保存原始image_id映射关系到数据集中
        for img in self._omni_api.dataset['images']:
            img['original_image_id'] = img['id']
        
        # 保存原始image_id映射关系到预测结果中
        self._logger.info("Processing predictions and updating image IDs...")
        processed_count = 0
        for pred in predictions:
            pred['original_image_id'] = pred['image_id']
            
            # 修改预测结果中的image_id
            if isinstance(pred['image_id'], str):
                file_name = pred['image_id'].split('/')[-1]  # 只取文件名部分
                if file_name in self.filename_to_id:
                    new_id = self.filename_to_id[file_name]
                    pred['image_id'] = new_id
                    
                    # 确保instances中的image_id与pred的image_id一致，同时保存原始id
                    for instance in pred['instances']:
                        instance['original_image_id'] = instance['image_id']
                        instance['image_id'] = new_id
            else:
                # 如果image_id已经是整数，确保instances使用相同的id
                for instance in pred['instances']:
                    instance['original_image_id'] = instance['image_id']
                    instance['image_id'] = pred['image_id']
        
            processed_count += 1
            if processed_count % 1000 == 0:
                self._logger.info(f"Processed {processed_count}/{len(predictions)} predictions")
        
        self._logger.info(f"Finished processing all {processed_count} predictions")
        
        omni_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        self._logger.info(f"Total number of instances after flattening: {len(omni_results)}")
        
        tasks = self._tasks or self._tasks_from_predictions(omni_results)
        self._logger.info(f"Tasks to evaluate: {tasks}")

        if self._metadata.name.endswith(("_novel", "_test")):
            category_path = "configs/category_meta.json" # TODO: hard coded
            self._logger.info(f"Loading category metadata from: {category_path}")
            metadata = util.load_json(category_path)
            omni3d_global_categories = metadata['thing_classes']
        else:
            omni3d_global_categories = MetadataCatalog.get('omni3d_model').thing_classes
        
        self._logger.info(f"Number of global categories: {len(omni3d_global_categories)}")

        # 过滤结果以只包含数据集中存在的类别
        dataset_results = []
        
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            self._logger.info(f"Number of classes in dataset: {num_classes}")

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            filtered_count = 0
            invalid_categories = set()
            
            # Before filtering, add debug information about the category mappings
            self._logger.info(f"Dataset thing classes: {self._metadata.thing_classes}")
            self._logger.info(f"Dataset ID to contiguous ID mapping: {dataset_id_to_contiguous_id}")
            self._logger.info(f"First few predictions category IDs: {[r['category_id'] for r in omni_results[:5]]}")

            # When filtering results, ensure proper category mapping
            for result in omni_results:
                category_id = result["category_id"]
                
                # Check if the category_id is already a dataset ID (not a contiguous ID)
                if category_id in dataset_id_to_contiguous_id:
                    # It's already a dataset ID, keep it as is
                    dataset_results.append(result)
                    filtered_count += 1
                    continue
                    
                # If it's a contiguous ID, try to map it
                if category_id < num_classes:
                    # Map contiguous ID to dataset ID
                    if category_id in reverse_id_mapping:
                        result["category_id"] = reverse_id_mapping[category_id]
                        dataset_results.append(result)
                        filtered_count += 1
                    else:
                        # Try to find the category by name
                        try:
                            cat_name = omni3d_global_categories[category_id]
                            if cat_name in self._metadata.thing_classes:
                                cat_idx = self._metadata.thing_classes.index(cat_name)
                                # Find the dataset ID for this category
                                for dataset_id, contiguous_id in dataset_id_to_contiguous_id.items():
                                    if contiguous_id == cat_idx:
                                        result["category_id"] = dataset_id
                                        dataset_results.append(result)
                                        filtered_count += 1
                                        break
                        except (IndexError, ValueError):
                            invalid_categories.add(category_id)
                else:
                    invalid_categories.add(category_id)
            
            if invalid_categories:
                self._logger.warning(f"Found predictions with invalid category IDs: {invalid_categories}")
            
            self._logger.info(f"Filtered results: {filtered_count} valid instances kept")

        # replace the results with the filtered instances
        omni_results = dataset_results

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "omni_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            
            self._logger.info(f"Final check before saving:")
            self._logger.info(f"Number of results to save: {len(omni_results)}")
            self._logger.info(f"Results type: {type(omni_results)}")
            
            if omni_results:
                self._logger.info(f"First result structure:")
                for key, value in omni_results[0].items():
                    self._logger.info(f"  {key}: {type(value)}")
                    if isinstance(value, (list, dict)):
                        self._logger.info(f"    Length/Size: {len(value)}")
            
            try:
                with PathManager.open(file_path, "w") as f:
                    json.dump(omni_results, f)
                    f.flush()
                    os.fsync(f.fileno())
                
                # 验证文件是否正确写入
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    self._logger.info(f"File successfully written, size: {file_size} bytes")
                    
                    # 读取文件验证内容
                    with PathManager.open(file_path, "r") as f:
                        saved_results = json.load(f)
                        self._logger.info(f"Successfully loaded saved results, count: {len(saved_results)}")
                        if len(saved_results) != len(omni_results):
                            self._logger.warning(f"Warning: Number of saved results ({len(saved_results)}) "
                                               f"differs from original ({len(omni_results)})")
                else:
                    self._logger.error("File was not created!")
            except Exception as e:
                self._logger.error(f"Error saving results: {str(e)}")
                import traceback
                self._logger.error(traceback.format_exc())

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox"}, f"Got unknown task: {task}!"
            
            if len(omni_results) == 0:
                self._logger.warning("No predictions found, skipping evaluation")
                # 当没有预测结果时，返回空的评估结果
                evals = {"2D": None, "3D": None}
                log_strs = {"2D": "", "3D": ""}
            else:
                # 检查GT数据集和预测结果的匹配情况
                gt_img_ids = set([img['id'] for img in self._omni_api.dataset.get('images', [])])
                pred_img_ids = set([r['image_id'] for r in omni_results])
                common_img_ids = gt_img_ids.intersection(pred_img_ids)
                
                self._logger.info(f"GT dataset has {len(gt_img_ids)} images")
                self._logger.info(f"Predictions cover {len(pred_img_ids)} images")
                self._logger.info(f"Common images between GT and predictions: {len(common_img_ids)}")
                
                gt_cat_ids = set([cat['id'] for cat in self._omni_api.dataset.get('categories', [])])
                pred_cat_ids = set([r['category_id'] for r in omni_results])
                common_cat_ids = gt_cat_ids.intersection(pred_cat_ids)
                
                self._logger.info(f"GT dataset has {len(gt_cat_ids)} categories")
                self._logger.info(f"Predictions cover {len(pred_cat_ids)} categories")
                self._logger.info(f"Common categories between GT and predictions: {len(common_cat_ids)}")
                
                if len(common_img_ids) == 0 or len(common_cat_ids) == 0:
                    self._logger.error("No common images or categories between GT and predictions!")
                    self._logger.info(f"GT image IDs (sample): {list(gt_img_ids)[:5]}")
                    self._logger.info(f"Pred image IDs (sample): {list(pred_img_ids)[:5]}")
                    self._logger.info(f"GT category IDs: {gt_cat_ids}")
                    self._logger.info(f"Pred category IDs: {pred_cat_ids}")
                
                evals, log_strs = _evaluate_predictions_on_omni(
                    self._omni_api,
                    omni_results,
                    task,
                    img_ids=img_ids,
                    only_2d=self._only_2d,
                    eval_prox=self._eval_prox,
                )

            modes = evals.keys() if evals else []
            for mode in modes:
                if evals[mode] is None:
                    continue
                res = self._derive_omni_results(
                    evals[mode],
                    task,
                    mode,
                    class_names=self._metadata.get("thing_classes"),
                )
                self._results[task + "_" + format(mode)] = res
                self._results[task + "_" + format(mode) + '_evalImgs'] = evals[mode].evalImgs
                self._results[task + "_" + format(mode) + '_evals_per_cat_area'] = evals[mode].evals_per_cat_area
                if mode == "3D":
                    self._results[task + "_" + format(mode) + '_nhd_accumulators'] = evals[mode].eval["nhd_accumulators"]
            
            if "2D" in log_strs:
                self._results["log_str_2D"] = log_strs["2D"]
            if "3D" in log_strs:
                self._results["log_str_3D"] = log_strs["3D"]

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(mode, ap=1, iouThr=None, areaRng="all", maxDets=100, log_str=""):
            p = self.params
            eval = self.eval

            if mode == "2D":
                iStr = (" {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}")

            elif mode == "3D":
                iStr = " {:<18} {} @[ IoU={:<9} | depth={:>6s} | maxDets={:>3d} ] = {:0.3f}"

            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"

            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:

                # dimension of precision: [TxRxKxAxM]
                s = eval["precision"]

                # IoU
                if iouThr is not None:
                    t = np.where(np.isclose(iouThr, p.iouThrs.astype(float)))[0]
                    s = s[t]

                s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
                
            else:
                mean_s = np.mean(s[s > -1])

            if log_str != "":
                log_str += "\n"

            log_str += "mode={} ".format(mode) + \
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            
            return mean_s, log_str

        def _summarizeDets(mode):

            params = self.params

            # the thresholds here, define the thresholds printed in `derive_omni_results`
            thres = [0.5, 0.75, 0.95] if mode == "2D" else [0.15, 0.25, 0.50]

            stats = np.zeros((13,))
            stats[0], log_str = _summarize(mode, 1)

            stats[1], log_str = _summarize(
                mode, 1, iouThr=thres[0], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[2], log_str = _summarize(
                mode, 1, iouThr=thres[1], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[3], log_str = _summarize(
                mode, 1, iouThr=thres[2], maxDets=params.maxDets[2], log_str=log_str
            )

            stats[4], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[5], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[6], log_str = _summarize(
                mode,
                1,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[7], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[0], log_str=log_str
            )

            stats[8], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[1], log_str=log_str
            )

            stats[9], log_str = _summarize(
                mode, 0, maxDets=params.maxDets[2], log_str=log_str
            )

            stats[10], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[1],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[11], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[2],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )

            stats[12], log_str = _summarize(
                mode,
                0,
                areaRng=params.areaRngLbl[3],
                maxDets=params.maxDets[2],
                log_str=log_str,
            )
            
            return stats, log_str

        if not self.eval:
            raise Exception("Please run accumulate() first")

        stats, log_str = _summarizeDets(self.mode)
        self.stats = stats

        return log_str


def calculate_nhd(pred_vertices, gt_vertices):
    """
    Calculate the Normalised Hungarian Distance (NHD) between predicted and ground truth box corners.
    Args:
        pred_vertices: (8, 3) numpy array of predicted box corners.
        gt_vertices: (8, 3) numpy array of ground truth box corners.
    Returns:
        NHD value.
    """
    # Calculate pairwise Euclidean distance between all corners
    cost_matrix = np.linalg.norm(pred_vertices[:, np.newaxis, :] - gt_vertices[np.newaxis, :, :], axis=2)
    
    # Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate NHD
    nhd = cost_matrix[row_ind, col_ind].sum()
    gt_diagonal = np.linalg.norm(gt_vertices.max(axis=0) - gt_vertices.min(axis=0))
    nhd /= gt_diagonal
    
    return nhd


def disentangled_nhd(pred_box, gt_box, components):
    """
    Calculate disentangled NHD for each component (xy, z, dimensions, pose).
    Args:
        pred_box: Dictionary containing predicted box parameters (xy, z, dimensions, pose).
        gt_box: Dictionary containing ground truth box parameters (xy, z, dimensions, pose).
        components: List of components to disentangle (e.g., ["xy", "z", "dimensions", "pose"]).
    Returns:
        Dictionary of disentangled NHD values for each component and overall NHD.
    """
    nhd_results = {}
    
    # Calculate un-disentangled NHD (overall NHD)
    pred_vertices, _ = get_cuboid_verts_faces(box3d=[pred_box['xy'][0], pred_box['xy'][1], pred_box['z'], *pred_box['dimensions']], R=pred_box['pose'])
    gt_vertices, _ = get_cuboid_verts_faces(box3d=[gt_box['xy'][0], gt_box['xy'][1], gt_box['z'], *gt_box['dimensions']], R=gt_box['pose'])
    
    # Convert vertices to numpy arrays
    pred_vertices = np.array(pred_vertices)
    gt_vertices = np.array(gt_vertices)
    
    nhd_results['overall'] = calculate_nhd(pred_vertices, gt_vertices)

    # Iterate over each component to calculate disentangled NHD
    for component in components:
        # Create a modified version of the predicted box that uses GT values for all but the current component
        modified_pred_box = pred_box.copy()
        for comp in components:
            if comp != component:
                modified_pred_box[comp] = gt_box[comp]
        
        # Get vertices of the modified predicted box and GT box
        pred_vertices, _ = get_cuboid_verts_faces(box3d=[modified_pred_box['xy'][0], modified_pred_box['xy'][1], modified_pred_box['z'], *modified_pred_box['dimensions']], R=modified_pred_box['pose'])
        gt_vertices, _ = get_cuboid_verts_faces(box3d=[gt_box['xy'][0], gt_box['xy'][1], gt_box['z'], *gt_box['dimensions']], R=gt_box['pose'])

        # Convert vertices to numpy arrays
        pred_vertices = np.array(pred_vertices)
        gt_vertices = np.array(gt_vertices)
        # Calculate NHD between modified prediction and GT
        nhd_results[component] = calculate_nhd(pred_vertices, gt_vertices)
    
    return nhd_results


class Omni3DevalWithNHD(Omni3Deval):
    def __init__(self, *args, iou_threshold_for_disentangled_metrics=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold_for_disentangled_metrics
        self._logger = logging.getLogger(__name__)
        self._predictions = []
        
    def evaluate(self):
        """
        Run per image evaluation on given images and store results in self.evalImgs
        """
        self._logger.info("Starting evaluation...")
        
        # 检查cocoDt中的预测结果而不是self._predictions
        if not self.cocoDt or len(self.cocoDt.getAnnIds()) == 0:
            self._logger.warning("No predictions found in cocoDt, skipping evaluation")
            self.evalImgs = {}
            return {}
            
        # 准备评估
        self._prepare()
        
        # 对每个图像进行评估
        self.evalImgs = defaultdict(list)  # 清除旧的评估结果
        self.eval = {}                     # 清除旧的评估结果
        self.stats = []                    # 清除旧的统计结果

        # 执行评估 - 使用父类的evaluateImg方法
        p = self.params
        catIds = p.catIds if p.useCats else [-1]
        computeIoU = self.computeIoU

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in catIds
        }

        maxDet = p.maxDets[-1]
        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        
        return self.eval

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Perform evaluation for single category and image with additional NHD metrics.
        """
        result = super().evaluateImg(imgId, catId, aRng, maxDet)
        if self.mode == "2D":
            return result
        if result is None:
            return None

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        
        if len(gt) == 0 or len(dt) == 0:
            return result

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]

        # Load IoUs computed during evaluation - use original format
        try:
            if self.mode == "2D":
                ious = self.ious2d[imgId, catId][0] if len(self.ious[imgId, catId][0]) > 0 else []
            else:
                ious = self.ious[imgId, catId][0] if len(self.ious[imgId, catId][0]) > 0 else []
        except (KeyError, IndexError):
            # Handle case where IoUs are not available
            self._logger.warning(f"IoUs not available for imgId={imgId}, catId={catId}")
            return result

        # Filter pairs based on IoU threshold - use original format with safety checks
        matched_pairs = []
        for dt_idx, dt_bbox in enumerate(dt):
            # Find the gt with the highest IoU for the current dt
            best_iou = 0
            best_gt_idx = -1
            
            try:
                for gt_idx, gt_bbox in enumerate(gt):
                    # Use original indexing style but with try/except for safety
                    try:
                        current_iou = ious[dt_idx, gt_idx]
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_gt_idx = gt_idx
                    except (IndexError, KeyError):
                        continue
                    
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    matched_pairs.append((dt[dt_idx], gt[best_gt_idx]))
            except Exception as e:
                self._logger.warning(f"Error matching detection {dt_idx}: {e}")
                continue

        # Calculate NHD and disentangled metrics for each matched pair
        components = ["xy", "z", "dimensions", "pose"]  # Use original components list
        nhd_metrics = []
        for dt_bbox, gt_bbox in matched_pairs:
            try:
                pred_box = {
                    "xy": dt_bbox["center_cam"][:2],
                    "z": dt_bbox["depth"],
                    "dimensions": dt_bbox["dimensions"],
                    "pose": dt_bbox["pose"]
                }
                gt_box = {
                    "xy": gt_bbox["center_cam"][:2],
                    "z": gt_bbox["depth"],
                    "dimensions": gt_bbox["dimensions"],
                    "pose": gt_bbox["R_cam"]
                }
                # Use original call with components parameter
                disentangled_results = disentangled_nhd(pred_box, gt_box, components)
                nhd_metrics.append(disentangled_results)
            except Exception as e:
                self._logger.warning(f"Error computing NHD metrics: {e}")
                continue

        result["nhd_metrics"] = nhd_metrics
        return result
    

    def accumulate(self, p=None):
        """
        Accumulate evaluation results by concatenating NHD metrics for the entire dataset.
        """
        # Call the parent class accumulate method
        super().accumulate(p)
        if self.mode == "2D":
            return 
        # Initialize accumulators for NHD metrics
        self.eval["nhd_accumulators"] = {"overall": [], "xy": [], "z": [], "dimensions": [], "pose": []}

        # Iterate over the evaluated images to collect NHD metrics
        for eval_img in self.evalImgs:
            if eval_img is None:
                continue
            if "nhd_metrics" in eval_img:
                for nhd_metric in eval_img["nhd_metrics"]:
                    for key in self.eval["nhd_accumulators"]:
                        if key in nhd_metric:
                            self.eval["nhd_accumulators"][key].append(nhd_metric[key])

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results, including average disentangled NHD.
        """
        # Call the parent class summarize method
        log_str = super().summarize()
        if self.mode == "2D":
            return log_str
        # Calculate average NHD for each component
        avg_nhd_results = {}
        for key, values in self.eval["nhd_accumulators"].items():
            if values:
                avg_nhd_results[key] = np.mean(values)
            else:
                avg_nhd_results[key] = float('nan')

        # Store the average NHD results in the evaluation summary
        self.eval["average_nhd"] = avg_nhd_results

        # Add the average disentangled NHD metrics to the log string
        if "average_nhd" in self.eval:
            log_str += "\nAverage Disentangled NHD Metrics:\n"
            for component, value in self.eval["average_nhd"].items():
                log_str += f"  {component}: {value:.4f}\n"

        return log_str