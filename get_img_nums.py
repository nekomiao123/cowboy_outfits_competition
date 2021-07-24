import random
import copy
import json
from pycocotools.coco import COCO

train_set = COCO('cowboydata/new_train.json')
train_img_ids = copy.copy(train_set.getImgIds())
print(len(train_img_ids))

val_set = COCO('cowboydata/new_valid.json')
val_img_ids = copy.copy(val_set.getImgIds())
print(len(val_img_ids))


