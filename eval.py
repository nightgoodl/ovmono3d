import json
import logging
from cubercnn.evaluation.omni3d_evaluation import Omni3DEvaluationHelper
from detectron2.data import MetadataCatalog
from cubercnn.data import simple_register
from detectron2.data import MetadataCatalog, DatasetCatalog

def evaluate_predictions_from_file(
    json_path,
    dataset_names,
    filter_settings,
    output_folder,
    eval_categories=None,
    category_mapping=None,
    use_category_names=False
):
    """
    直接从JSON文件读取预测结果并进行评估
    
    Args:
        json_path (str): omni_instances_results.json的路径
        dataset_names (list[str]): 要评估的数据集名称列表
        filter_settings (dict): 数据集过滤设置
        output_folder (str): 输出文件夹路径
        eval_categories (set): 要评估的类别集合
        category_mapping (dict): 类别ID映射字典，将预测结果中的类别ID映射到数据集中的类别ID
        use_category_names (bool): 是否使用类别名称进行匹配而不是ID
    """
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 读取预测结果
    logger.info(f"Loading predictions from {json_path}")
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # 检查预测结果格式
    if predictions and isinstance(predictions, list):
        # 检查第一个元素是否有'instances'键
        if predictions and isinstance(predictions[0], dict) and 'instances' not in predictions[0]:
            logger.info("Detected flat prediction format without 'instances' key")
            logger.info(f"Sample prediction: {predictions[0]}")
            
            # 将扁平格式转换为带有instances的格式
            # 按image_id分组
            predictions_by_image = {}
            for pred in predictions:
                image_id = pred.get('image_id')
                if image_id not in predictions_by_image:
                    predictions_by_image[image_id] = {
                        'image_id': image_id,
                        'instances': []
                    }
                predictions_by_image[image_id]['instances'].append(pred)
            
            # 转换为列表
            predictions = list(predictions_by_image.values())
            logger.info(f"Converted to {len(predictions)} image predictions with instances")
    
    # 创建评估器
    evaluator = Omni3DEvaluationHelper(
        dataset_names=dataset_names,
        filter_settings=filter_settings,
        output_folder=output_folder,
        eval_categories=eval_categories
    )

    # 按数据集组织预测结果
    predictions_by_dataset = {dataset: [] for dataset in dataset_names}
    
    # 创建image_id到dataset的映射
    image_to_dataset = {}
    for dataset_name in dataset_names:
        dataset = DatasetCatalog.get(dataset_name)
        for img in dataset:
            image_to_dataset[img['image_id']] = dataset_name

    # 获取数据集的类别映射信息
    dataset_category_maps = {}
    
    # 如果提供了自定义映射，使用它
    if category_mapping:
        logger.info("Using custom category mapping")
        logger.info(f"Category mapping: {category_mapping}")
    else:
        # 如果没有提供映射，尝试从元数据中获取
        logger.info("No custom mapping provided, attempting to create mapping from metadata")
        for dataset_name in dataset_names:
            metadata = MetadataCatalog.get(dataset_name)
            
            # 检查是否有thing_classes和thing_dataset_id_to_contiguous_id
            if hasattr(metadata, 'thing_classes') and hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
                # 创建类别名称到ID的映射
                category_map = {name: i for i, name in enumerate(metadata.thing_classes)}
                dataset_category_maps[dataset_name] = {
                    'name_to_id': category_map,
                    'dataset_to_contiguous': metadata.thing_dataset_id_to_contiguous_id,
                    'contiguous_to_dataset': {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
                }
                logger.info(f"Created category mapping for {dataset_name}")
                logger.info(f"  Thing classes: {metadata.thing_classes}")
                logger.info(f"  Dataset ID to contiguous ID: {metadata.thing_dataset_id_to_contiguous_id}")
    
    # 检查预测结果中的类别ID
    pred_category_ids = set()
    for pred in predictions:
        for instance in pred.get('instances', []):
            if 'category_id' in instance:
                pred_category_ids.add(instance['category_id'])
    
    logger.info(f"Prediction category IDs: {pred_category_ids}")
    
    # 如果使用类别名称进行匹配，创建ID到名称的映射
    id_to_name_mapping = {}
    name_to_id_mapping = {}
    
    if use_category_names:
        logger.info("Using category names for matching")
        # 假设预测ID是从0开始的连续整数，对应于模型的类别列表
        model_categories = list(eval_categories)  # 使用评估类别作为模型类别
        logger.info(f"Model categories (in order): {model_categories}")
        
        for i, cat_name in enumerate(model_categories):
            id_to_name_mapping[i] = cat_name
        
        # 为每个数据集创建名称到ID的映射
        for dataset_name in dataset_names:
            metadata = MetadataCatalog.get(dataset_name)
            if hasattr(metadata, 'thing_classes') and hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
                for dataset_id, contiguous_id in metadata.thing_dataset_id_to_contiguous_id.items():
                    if contiguous_id < len(metadata.thing_classes):
                        cat_name = metadata.thing_classes[contiguous_id]
                        name_to_id_mapping[cat_name] = dataset_id
        
        logger.info(f"Created ID to name mapping: {id_to_name_mapping}")
        logger.info(f"Created name to ID mapping: {name_to_id_mapping}")
        
        # 创建基于名称的类别映射
        name_based_mapping = {}
        for pred_id, cat_name in id_to_name_mapping.items():
            if cat_name in name_to_id_mapping:
                name_based_mapping[pred_id] = name_to_id_mapping[cat_name]
        
        if name_based_mapping:
            logger.info(f"Created name-based category mapping: {name_based_mapping}")
            category_mapping = name_based_mapping
    
    # 检查数据集中的类别ID
    for dataset_name in dataset_names:
        metadata = MetadataCatalog.get(dataset_name)
        if hasattr(metadata, 'thing_dataset_id_to_contiguous_id'):
            dataset_ids = set(metadata.thing_dataset_id_to_contiguous_id.keys())
            logger.info(f"Dataset {dataset_name} category IDs: {dataset_ids}")
            
            # 检查是否需要自动创建映射
            if not category_mapping and not pred_category_ids.intersection(dataset_ids) and len(pred_category_ids) > 0:
                logger.info(f"No overlap between prediction category IDs and dataset category IDs for {dataset_name}")
                logger.info("Attempting to create automatic mapping...")
                
                # 如果预测ID是连续的整数（0, 1, 2...），可能是模型输出的contiguous ID
                if all(isinstance(id, int) for id in pred_category_ids) and max(pred_category_ids) < len(metadata.thing_classes):
                    # 创建从contiguous ID到dataset ID的映射
                    auto_mapping = {}
                    for pred_id in pred_category_ids:
                        if pred_id < len(metadata.thing_classes):
                            # 找到对应的dataset ID
                            cat_name = metadata.thing_classes[pred_id]
                            for dataset_id, contiguous_id in metadata.thing_dataset_id_to_contiguous_id.items():
                                if contiguous_id == pred_id:
                                    auto_mapping[pred_id] = dataset_id
                                    break
                    
                    if auto_mapping:
                        logger.info(f"Created automatic mapping: {auto_mapping}")
                        category_mapping = auto_mapping

    # 将预测结果分配到对应的数据集，并应用类别映射
    for pred in predictions:
        dataset = image_to_dataset.get(pred['image_id'])
        if dataset:
            # 创建预测结果的副本，以便修改类别ID
            pred_copy = pred.copy()
            pred_copy['instances'] = []
            
            # 确保pred有'instances'键
            instances = pred.get('instances', [])
            # 如果instances为空且pred本身可能是一个实例，则将其视为单个实例
            if not instances and 'category_id' in pred:
                instances = [pred]
            
            for instance in instances:
                # 创建实例的副本
                instance_copy = instance.copy()
                
                # 应用类别ID映射
                if category_mapping and 'category_id' in instance_copy:
                    original_id = instance_copy['category_id']
                    if original_id in category_mapping:
                        instance_copy['category_id'] = category_mapping[original_id]
                        logger.debug(f"Mapped category ID {original_id} to {instance_copy['category_id']}")
                
                pred_copy['instances'].append(instance_copy)
            
            predictions_by_dataset[dataset].append(pred_copy)

    # 对每个数据集进行评估
    for dataset_name in dataset_names:
        logger.info(f"\nEvaluating dataset: {dataset_name}")
        dataset_predictions = predictions_by_dataset[dataset_name]
        logger.info(f"Found {len(dataset_predictions)} predictions for this dataset")
        
        # 添加预测结果到评估器
        evaluator.add_predictions(dataset_name, dataset_predictions)
        
        # 执行评估
        evaluator.evaluate(dataset_name)

    # 汇总所有数据集的结果
    logger.info("\nGenerating summary for all datasets...")
    evaluator.summarize_all()

    return evaluator.results_analysis, evaluator.results_omni3d

# 使用示例:
if __name__ == "__main__":
    # 配置参数
    json_path = "../../output/ovmono3d_depth/inference/iter_base/Objectron_test/omni_instances_results.json"
    dataset_names = ["Objectron_test"]  # 根据实际情况修改
    
    # 过滤设置示例
    filter_settings = {
        'category_names': ["bottle", "bowl", "camera", "can", "laptop", "mug"],  # 需要评估的类别
        'ignore_names': [],  # 需要忽略的类别
        'truncation_thres': 0.99,  # 截断阈值
        'visibility_thres': 0.01,  # 可见度阈值
        'min_height_thres': 0.00,  # 最小高度阈值
        'max_height_thres': 1.50,  # 最大高度阈值
        'modal_2D_boxes': False,  # 是否使用modal 2D边界框
        'trunc_2D_boxes': False,  # 是否使用截断的2D边界框
        'max_depth': 1e8,  # 最大深度
    }
    
    output_folder = "evaluation_results"
    
    # 要评估的类别
    eval_categories = {
       'bicycle', 'books', 'bottle', 'camera', 'cereal box', 'chair', 'cup', 'laptop', 'shoes'
    }  # 根据实际情况修改
    
    # 类别ID映射示例 - 将预测结果中的类别ID映射到数据集中的类别ID
    # 例如，如果预测结果中bottle的ID是1，但在数据集中是0
    category_mapping = {
        1: 0,  # bottle
        2: 1,  # bowl
        3: 2,  # camera
        # 根据实际情况添加更多映射
    }
    
    # 运行评估
    results_analysis, results_omni3d = evaluate_predictions_from_file(
        json_path,
        dataset_names,
        filter_settings,
        output_folder,
        eval_categories,
        category_mapping=None,  # 设置为None以使用基于名称的自动映射
        use_category_names=True  # 启用基于类别名称的匹配
    )
    
    # 打印结果
    print("\nAnalysis Results:")
    print(results_analysis)
    print("\nOmni3D Results:")
    print(results_omni3d)