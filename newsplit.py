import random
import copy
import json
from pycocotools.coco import COCO

def create_subset(c, cats, test_n=180):
    new_coco = {}
    new_coco['info'] = {"description": "CowboySuit",
                        "url": "http://github.com/dmlc/gluon-cv",
                        "version": "1.0","year": 2021,
                        "contributor": "GluonCV/AutoGluon",
                        "date_created": "2021/07/01"}
    new_coco["licenses"]: [
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}]
    cat_ids = c.getCatIds(cats)
    train_img_ids = set()

    test_img_ids = set()
    for cat in cat_ids[::-1]:
        img_ids = copy.copy(c.getImgIds(catIds=[cat]))
        random.shuffle(img_ids)
        tn = min(test_n, int(len(img_ids) * 0.5))
        new_test = set(img_ids[:tn])
        exist_test_ids = new_test.intersection(train_img_ids)
        test_ids = new_test.difference(exist_test_ids)
        train_ids = set(img_ids).difference(test_ids)
        print(tn, len(img_ids), len(new_test), len(test_ids), len(train_ids))
        train_img_ids.update(train_ids)
        test_img_ids.update(test_ids)
        print(len(test_img_ids))

    # prune duplicates
    dup = train_img_ids.intersection(test_img_ids)
    train_img_ids = train_img_ids - dup

    train_anno_ids = set()
    test_anno_ids = set()
    for cat in cat_ids:
        train_anno_ids.update(c.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
        test_anno_ids.update(c.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))

    assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(train_img_ids.intersection(test_img_ids))
    assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts'
    print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
    print('test img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
    new_coco_test = copy.deepcopy(new_coco)

    new_coco["images"] = c.loadImgs(list(train_img_ids))
    new_coco["annotations"] = c.loadAnns(list(train_anno_ids))
    for ann in new_coco["annotations"]:
        ann.pop('segmentation', None)
    new_coco["categories"] = c.loadCats(cat_ids)

    new_coco_test["images"] = c.loadImgs(list(test_img_ids))
    new_coco_test["annotations"] = c.loadAnns(list(test_anno_ids))
    for ann in new_coco_test["annotations"]:
        ann.pop('segmentation', None)
    new_coco_test["categories"] = c.loadCats(cat_ids)
    print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
    print('new test split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))
    return new_coco, new_coco_test


coco = COCO('cowboydata/train.json')
print('begin split!!!!!!!!!!!!!!!!!!!!!!!!!!')

nc, nc_test = create_subset(coco, ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket', ])

with open('cowboydata/new_train.json', 'w') as f:
    json.dump(nc, f)

with open('cowboydata/new_valid.json', 'w') as f:
    json.dump(nc_test, f)
print("Down!!!!!!!!!!!!!!!!!")

