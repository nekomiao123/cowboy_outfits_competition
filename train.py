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

# baseline_cfg_path = "../mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py"
baseline_cfg_path = "../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py"
cfg = Config.fromfile(baseline_cfg_path)

'''
General Training Settings
'''
model_name = 'yolov3_d53_mstrain-608_273e'
fold = 0
job = 1

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
# for head in cfg.model.roi_head.bbox_head:
#     head.num_classes = 5
# cfg.model.roi_head.bbox_head.num_classes = 5
cfg.model.bbox_head.num_classes = 5

cfg.gpu_ids = [4]

# Setting pretrained model in the init_cfg which is required 
# for transfer learning as per the latest MMdetection update
# cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
# cfg.model.backbone.init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')
# cfg.model.pop('pretrained', None)

cfg.runner.max_epochs = 12 # Epochs for the runner that runs the workflow 
cfg.total_epochs = 12

# Learning rate of optimizers. The LR is divided by 8 since the config file is originally for 8 GPUs
cfg.optimizer.lr = 0.02/8

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
cfg.data_root = '/cowboydata'

cfg.data.train.img_prefix = './cowboydata/images' # Prefix of image path
cfg.data.train.classes = cfg.classes
cfg.data.train.ann_file = './cowboydata/train.json'
cfg.data.train.type='CocoDataset'

cfg.data.val.img_prefix = './cowboydata/images' # Prefix of image path
cfg.data.val.classes = cfg.classes
cfg.data.val.ann_file = './cowboydata/train.json'
cfg.data.val.type='CocoDataset'

cfg.data.test.img_prefix = './cowboydata/images' # Prefix of image path
cfg.data.test.classes = cfg.classes
cfg.data.test.ann_file =  './cowboydata/train.json'
cfg.data.test.type='CocoDataset'

cfg.data.samples_per_gpu = 8 # Batch size of a single GPU used in testing
cfg.data.workers_per_gpu = 4 # Worker to pre-fetch data for each single GPU

'''
Setting Metric for Evaluation
'''

cfg.evaluation.metric = 'bbox' # Metrics used during evaluation

# Set the epoch intervel to perform evaluation
cfg.evaluation.interval = 1

# Set the iou threshold of the mAP calculation during evaluation
cfg.evaluation.iou_thrs = [0.5]

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

model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
model.init_weights()
datasets = [build_dataset(cfg.data.train)]
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
