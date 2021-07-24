import sys
import os
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# Check MMDetection installation
from mmdet.apis import set_random_seed

# Imports
import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

import random
import numpy as np
from pathlib import Path

global_seed = 111

def set_seed(seed=global_seed):
    """Sets the random seeds."""
    set_random_seed(seed, deterministic=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

'''
Prepare the MMDetection Config
'''
from mmcv import Config

baseline_cfg_path = "../mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py"
# baseline_cfg_path = "../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py"

cfg = Config.fromfile(baseline_cfg_path)

'''
General Training Settings
'''
model_name = 'cascade_rcnn_x101_32x4d_fpn_1x'
fold = 0
job = 7

# Folder to store model logs and weight files
job_folder = f'./working/job{job}_{model_name}_fold{fold}'
cfg.work_dir = job_folder

# Change the wnd username and project name below
wnb_username = 'nekokiku'
wnb_project_name = 'cow-boy-detection'

# Set seed thus the results are more reproducible
cfg.seed = global_seed

if not os.path.exists(job_folder):
    os.makedirs(job_folder)

print("Job folder:", job_folder)

# Set the number of classes
for head in cfg.model.roi_head.bbox_head:
    head.num_classes = 5
# cfg.model.roi_head.bbox_head.num_classes = 5
# cfg.model.bbox_head.num_classes = 5

# cfg.gpu_ids = [5]
cfg.fp16 = dict(loss_scale='dynamic')

# Setting pretrained model in the init_cfg which is required 
# for transfer learning as per the latest MMdetection update
cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')
cfg.model.pop('pretrained', None)

cfg.runner.max_epochs = 12 # Epochs for the runner that runs the workflow 
cfg.total_epochs = 12

# Learning rate of optimizers. The LR is divided by 8 since the config file is originally for 8 GPUs
cfg.optimizer.lr = 0.02/4

## Learning rate scheduler config used to register LrUpdater hook
cfg.lr_config = dict(
    policy='CosineAnnealing', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    by_epoch=False,
    warmup='linear', # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500, # The number of iterations for warmup
    warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
    min_lr=1e-07)

# config to register logger hook
cfg.log_config.interval = 40 # Interval to print the log

# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
cfg.checkpoint_config.interval = 1 # The save interval is 1

'''
Dataset
'''

cfg.dataset_type = 'CocoDataset' # Dataset type, this will be used to define the dataset
cfg.classes = ("belt","sunglasses","boot","cowboy_hat","jacket")

cfg.data_root = '/home/tantianlong/.code/code/cowboy/cowboydata'

cfg.data.train.img_prefix = cfg.data_root + '/images' # Prefix of image path
cfg.data.train.classes = cfg.classes
cfg.data.train.ann_file = cfg.data_root + '/new_train.json'
cfg.data.train.type='CocoDataset'

cfg.data.val.img_prefix = cfg.data_root + '/images' # Prefix of image path
cfg.data.val.classes = cfg.classes
cfg.data.val.ann_file = cfg.data_root + '/new_valid.json'
cfg.data.val.type='CocoDataset'

cfg.data.test.img_prefix = cfg.data_root + '/images' # Prefix of image path
cfg.data.test.classes = cfg.classes
cfg.data.test.ann_file =  cfg.data_root + '/new_valid.json'
cfg.data.test.type='CocoDataset'

cfg.data.samples_per_gpu = 4 # Batch size of a single GPU used in testing
cfg.data.workers_per_gpu = 4 # Worker to pre-fetch data for each single GPU

'''
Setting Metric for Evaluation
'''

cfg.evaluation.metric = 'bbox' # Metrics used during evaluation

# Set the epoch intervel to perform evaluation
cfg.evaluation.interval = 1

# Set the iou threshold of the mAP calculation during evaluation
# cfg.evaluation.iou_thrs = [0.75]

cfg.evaluation.save_best='bbox_mAP'
'''
Prepare the Pre-processing & Augmentation Pipelines
'''

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.15, rotate_limit=15, p=0.4),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.5),
    dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
#     dict(type='MixUp', p=0.2, lambd=0.5),
    dict(type="Blur", p=1.0, blur_limit=7),
    dict(type='CLAHE', p=0.5),
    dict(type='Equalize', mode='cv', p=0.4),
    dict(
        type="OneOf",
        transforms=[
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.4,
    ),

#     dict(type='MixUp', p=0.2, lambd=0.5),
#     dict(type='RandomRotate90', p=0.5),
#     dict(type='CLAHE', p=0.5),
#     dict(type='InvertImg', p=0.5),
#     dict(type='Equalize', mode='cv', p=0.4),
#     dict(type='MedianBlur', blur_limit=3, p=0.1)
    ]


cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


'''
About wandb
'''

cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                        dict(type='WandbLoggerHook',
                             init_kwargs=dict(project=wnb_project_name,
                                              name=f'exp-{model_name}-fold{fold}-job{job}',
                                              entity=wnb_username))
                       ]

'''
Save Config File
'''

cfg.dump(F'{cfg.work_dir}/my_config.py')

'''
Build Dataset and Start Training
'''

# model = build_detector(cfg.model,
#                        train_cfg=cfg.get('train_cfg'),
#                        test_cfg=cfg.get('test_cfg'))
# model.init_weights()
# datasets = [build_dataset(cfg.data.train)]
# train_detector(model, datasets[0], cfg, distributed=False, validate=True)
