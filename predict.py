from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
from tqdm import tqdm
import json  # for dumping json serialized results
import zipfile  # for creating submission zip file
import pandas as pd

def create_submission(df, model, score_thresh=0.5):
    results = []
    for index, row in tqdm(df.iterrows()):
        img_id = row['id']
        file_name = row['file_name']
        img_base = './cowboydata/images/'
        img = img_base + file_name
        result = inference_detector(model, img)

        for i in range(5):
            if len(result[i]) != 0:
                for j in result[i]:
                    j = np.array(j).tolist()
                    if j[-1] >= score_thresh:
                        # 这里注意原来是xmin, ymin, xmax, ymax.
                        # coco 需要的数据格式是xmin, ymin, w, h.
                        pred = {'image_id': img_id,
                                'category_id': int(classes_id[i]),
                                'bbox': [j[0], j[1], j[2]-j[0], j[3]-j[1]],
                                'score': j[-1]}
                        results.append(pred)
    return results

# base name 
base_name = 'answer'
zip_name = 'cascade_job7'
# classes
classes = ('belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket')
classes_id = ('87', '1034', '131', '318', '588')
# Choose to use a config and checkpoint
config = './working/job7_cascade_rcnn_x101_32x4d_fpn_1x_fold0/my_config.py'
# Setup a checkpoint file to load
checkpoint = './working/job7_cascade_rcnn_x101_32x4d_fpn_1x_fold0/epoch_6.pth'
# val path
val_path = './cowboydata/valid.csv'
# submission base
submission_base = './submission/'
# submission name
submission_name = base_name + '.json'
# submission path
submission_path = submission_base + submission_name
# zipfile name
zipfile_name = submission_base + 'zip_'+ zip_name +'.zip'

model = init_detector(config, checkpoint, device='cuda:5')
submission_df = pd.read_csv(val_path)
submission = create_submission(submission_df, model)

print(config)
print(checkpoint)
with open(submission_path, 'w') as f:
    json.dump(submission, f)
zf = zipfile.ZipFile(zipfile_name, 'w')
zf.write(submission_path, submission_name)
zf.close()
