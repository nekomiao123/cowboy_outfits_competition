import random
import copy
import json
from pycocotools.coco import COCO

def create_subset(c, cats, test_n=20):
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
    valid_img_ids = set()
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
        if len(train_ids) > 2000:
            r = 2
            cut = len(train_ids) // r
            valid_ids = list(train_ids)[cut:(cut + 300)]
            train_ids = list(train_ids)[:cut]
        else:
            r = 2
            cut = (len(train_ids) // r)
            max_cut = min(len(train_ids), cut + 200)
            valid_ids = list(train_ids)[cut:max_cut]
            train_ids = list(train_ids)[:cut]
        train_img_ids.update(train_ids)
        valid_img_ids.update(valid_ids)
        test_img_ids.update(test_ids)


    # prune duplicates
    dup = train_img_ids.intersection(test_img_ids)
    train_img_ids = train_img_ids - dup
    dup1 = train_img_ids.intersection(valid_img_ids)
    train_img_ids = train_img_ids - dup1
    dup2 = valid_img_ids.intersection(test_img_ids)
    valid_img_ids = valid_img_ids - dup2
    train_anno_ids = set()
    valid_anno_ids = set()
    test_anno_ids = set()
    for cat in cat_ids:
        train_anno_ids.update(c.getAnnIds(imgIds=list(train_img_ids), catIds=[cat]))
        valid_anno_ids.update(c.getAnnIds(imgIds=list(valid_img_ids), catIds=[cat]))
        test_anno_ids.update(c.getAnnIds(imgIds=list(test_img_ids), catIds=[cat]))

    assert len(train_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(train_img_ids.intersection(test_img_ids))
    assert len(train_img_ids.intersection(valid_img_ids)) == 0, 'img id conflicts, {} '.format(valid_img_ids.intersection(train_img_ids))
    assert len(valid_img_ids.intersection(test_img_ids)) == 0, 'img id conflicts, {} '.format(valid_img_ids.intersection(test_img_ids))
    assert len(train_anno_ids.intersection(test_anno_ids)) == 0, 'anno id conflicts'
    print('train img ids #:', len(train_img_ids), 'train anno #:', len(train_anno_ids))
    print('valid img ids #:', len(valid_img_ids), 'valid anno #:', len(valid_anno_ids))
    print('test img ids #:', len(test_img_ids), 'test anno #:', len(test_anno_ids))
    new_coco_test = copy.deepcopy(new_coco)
    new_coco_valid = copy.deepcopy(new_coco)

    # combine train and valid data set
    train_img_ids = train_img_ids | valid_img_ids
    train_anno_ids = train_anno_ids | valid_anno_ids

    new_coco["images"] = c.loadImgs(list(train_img_ids))
    new_coco["annotations"] = c.loadAnns(list(train_anno_ids))
    for ann in new_coco["annotations"]:
        ann.pop('segmentation', None)
    new_coco["categories"] = c.loadCats(cat_ids)

    new_coco_valid["images"] = c.loadImgs(list(valid_img_ids))
    new_coco_valid["annotations"] = c.loadAnns(list(valid_anno_ids))
    for ann in new_coco_valid["annotations"]:
        ann.pop('segmentation', None)
    new_coco_valid["categories"] = c.loadCats(cat_ids)

    new_coco_test["images"] = c.loadImgs(list(test_img_ids))
    new_coco_test["annotations"] = c.loadAnns(list(test_anno_ids))
    for ann in new_coco_test["annotations"]:
        ann.pop('segmentation', None)
    new_coco_test["categories"] = c.loadCats(cat_ids)
    print('new train split, images:', len(new_coco["images"]), 'annos:', len(new_coco["annotations"]))
    print('new valid split, images:', len(new_coco_valid["images"]), 'annos:', len(new_coco_valid["annotations"]))
    print('new test split, images:', len(new_coco_test["images"]), 'annos:', len(new_coco_test["annotations"]))
    return new_coco, new_coco_valid, new_coco_test


coco = COCO('train.json')
print('begin split!!!!!!!!!!!!!!!!!!!!!!!!!!')

nc, nc_valid, nc_test = create_subset(coco, ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket', ])

with open('new_train.json', 'w') as f:
    json.dump(nc, f)

with open('new_valid.json', 'w') as f:
    json.dump(nc_test, f)

