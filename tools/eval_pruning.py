import sys
sys.path.append('./')
from ultralytics import YOLO
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# kjg: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_kjg/cls1_coco_20231016_v0.2/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_kjg/cls1_20230906_v0.2.yaml
# hw: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_hw/cls2_rdd_20231016_v0.4_canada/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml
# rdd: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/rdd2020/cls4_20231016/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/rdd2020_cls4.yaml

weight_path = "/home/ubuntu/data1/lorenzo/myrepo/ultralytics/runs/bt_hw/yolov8m_pruning0.5_cls2_coco_20231026_v0.4_canada/step_15_finetune/weights/best.pt"
data_path = "/home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml"
imgsz = 320
batch = 32
conf = 0.45
iou = 0.6

model = YOLO(weight_path)
metrics = model.val(data=data_path, 
                    imgsz=imgsz,
                    batch=batch,
                    device=1,
                    # save_json=True,
                    # save_txt=True,
                    # save_conf=True,
                    conf=conf,
                    iou=iou)


print(f"map50@95: {metrics.box.map}")
print(f"map50: {metrics.box.map50}")
print(metrics.box.maps)