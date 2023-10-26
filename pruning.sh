python tools/yolov8_pruning.py \
      --model runs/bt_hw/yolov8m_cls2_coco_20231026_v0.4_canada/weights/best.pt \
      --cfg ultralytics/yolo/cfg/default.yaml \
      --data /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml \
      --batch-size 64 \
      --imgsz 320 \
      --project runs/bt_hw/yolov8m_pruning0.5_cls2_coco_20231026_v0.4_canada