# Get the best epoch number
import json
import numpy as np
from collections import defaultdict

log_file = './working/job4_cascade_rcnn_x101_32x4d_fpn_1x_fold0/20210723_182039.log.json'

# Source: mmdetection/tools/analysis_tools/analyze_logs.py 
def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts

log_dict = load_json_logs([log_file])
best_epoch = np.argmax([item['bbox_mAP'][0] for item in log_dict[0].values()])+1
best_mAP = np.max([item['bbox_mAP'][0] for item in log_dict[0].values()])

print(f'The best mAP is {best_mAP} and the epoch is {best_epoch}')
