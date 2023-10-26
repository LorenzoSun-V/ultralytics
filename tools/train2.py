import sys
sys.path.append('./')
from ultralytics import YOLO

model = YOLO("yolov8s.yaml")
model = YOLO("/home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/rdd2020/cls4_20231016/weights/best.pt")

# hw
model.train(data="/home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_kjg/cls1_20230906_v0.2.yaml", 
            epochs=100, 
            imgsz=640,
            batch=16,
            device=1,
            project="runs/bt_kjg",
            name="cls1_rdd_20231016_v0.2",
            exist_ok=True,
)