import sys
import argparse
import os
import struct
import torch

import sys
sys.path.append('./')

pt_file = "/home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_kjg/cls1_coco_20231016_v0.2/weights/best.pt"
wts_file = "/home/ubuntu/Downloads/kjg_cls1_20230906_v0.2/models/yolov8_cls1_coco_20230906_v0.2.wts"

# Initialize
device = 'cpu'

# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]

delattr(model.model[-1], 'anchors')

model.to(device).eval()

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')