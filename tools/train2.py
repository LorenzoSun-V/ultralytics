import sys
sys.path.append('./')
from ultralytics import YOLO

model = YOLO("yolov8s.yaml")
model = YOLO("/home/ubuntu/data1/lorenzo/Detection/ultralytics/yolov8s.pt")

# hw
model.train(data="/home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml", 
            epochs=100, 
            imgsz=320,
            batch=64,
            device=1,
            project="runs/bt_hw",
            name="yolov8s_cls2_coco_20231026_v0.4_canada",
            exist_ok=True,
            patience=20,
)