import os
import sys
import random
import numpy as np
from pathlib import Path

import torch, torchvision

import mmdet
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

from utils import set_seed

'''
Set global seed
'''
global_seed = 1234
set_seed(global_seed)

'''
Prepare the MMDetection Config
'''
# baseline_cfg_path = "../mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py"
# baseline_cfg_path = "../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py"

baseline_cfg_path = "../mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(baseline_cfg_path)

'''
General Training Settings
'''
model_name = 'cascade_rcnn_r50_fpn_1x'
job = 2

# Folder to store model logs and weight files
job_folder = f'./working/job{job}_{model_name}'
cfg.work_dir = job_folder

# Change the wnd username and project name below
wnb_username = 'nekokiku'
wnb_project_name = 'cow-boy-detection'

# Set seed thus the results are more reproducible
cfg.seed = global_seed
# You should change this if you use different model
cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

if not os.path.exists(job_folder):
    os.makedirs(job_folder)

print("Job folder:", job_folder)

# Set the number of classes
for head in cfg.model.roi_head.bbox_head:
    head.num_classes = 5
# cfg.model.roi_head.bbox_head.num_classes = 5
# cfg.model.bbox_head.num_classes = 5

# cfg.gpu_ids = [6]
cfg.fp16 = dict(loss_scale='dynamic')

# Setting pretrained model in the init_cfg which is required 
# for transfer learning as per the latest MMdetection update
# cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
# cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')
# cfg.model.pop('pretrained', None)

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

cfg.data_root = '/home/listu/code/learn/cowboy/cowboy_outfits_competition/cowboydata'

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

cfg.data.samples_per_gpu = 8 # Batch size of a single GPU used in testing
cfg.data.workers_per_gpu = 4 # Worker to pre-fetch data for each single GPU

'''
Setting Metric for Evaluation
'''

cfg.evaluation.metric = 'bbox' # Metrics used during evaluation

# Set the epoch intervel to perform evaluation
cfg.evaluation.interval = 1

cfg.evaluation.save_best='bbox_mAP'
'''
Prepare the Pre-processing & Augmentation Pipelines
'''

# albu_train_transforms = [
#     dict(type='ShiftScaleRotate', shift_limit=0.0625,
#          scale_limit=0.15, rotate_limit=15, p=0.4),
#     dict(type='RandomBrightnessContrast', brightness_limit=0.2,
#          contrast_limit=0.2, p=0.5),
#     dict(type='IAAAffine', shear=(-10.0, 10.0), p=0.4),
# #     dict(type='MixUp', p=0.2, lambd=0.5),
#     dict(type="Blur", p=1.0, blur_limit=7),
#     dict(type='CLAHE', p=0.5),
#     dict(type='Equalize', mode='cv', p=0.4),
#     dict(
#         type="OneOf",
#         transforms=[
#             dict(type="GaussianBlur", p=1.0, blur_limit=7),
#             dict(type="MedianBlur", p=1.0, blur_limit=7),
#         ],
#         p=0.4,
#     ),

# #     dict(type='MixUp', p=0.2, lambd=0.5),
# #     dict(type='RandomRotate90', p=0.5),
# #     dict(type='CLAHE', p=0.5),
# #     dict(type='InvertImg', p=0.5),
# #     dict(type='Equalize', mode='cv', p=0.4),
# #     dict(type='MedianBlur', blur_limit=3, p=0.1)
#     ]


# cfg.train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#         type='BboxParams',
#         format='pascal_voc',
#         label_fields=['gt_labels'],
#         min_visibility=0.0,
#         filter_lost_elements=True),
#         keymap=dict(img='image', gt_bboxes='bboxes'),
#         update_pad_shape=False,
#         skip_img_without_anno=True),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]

'''
About wandb
'''

cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                        dict(type='WandbLoggerHook',
                             init_kwargs=dict(project=wnb_project_name,
                                              name=f'exp-{model_name}-job{job}',
                                              entity=wnb_username))
                       ]

'''
Save Config File
'''
cfg_path = f'{job_folder}/job{job}_{Path(baseline_cfg_path).name}'
print(cfg_path)

# Save config file for inference later
cfg.dump(cfg_path)
# print(f'Config:\n{cfg.pretty_text}')

'''
Build Dataset and Start Training
'''

# model = build_detector(cfg.model,
#                        train_cfg=cfg.get('train_cfg'),
#                        test_cfg=cfg.get('test_cfg'))
# model.init_weights()
# datasets = [build_dataset(cfg.data.train)]
# train_detector(model, datasets[0], cfg, distributed=False, validate=True)
