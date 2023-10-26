import sys
sys.path.append('./')
from ultralytics import YOLO


# kjg: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_kjg/cls1_coco_20231016_v0.2/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_kjg/cls1_20230906_v0.2.yaml
# hw: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_hw/cls2_rdd_20231016_v0.4_canada/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml
# rdd: /home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/rdd2020/cls4_20231016/weights/best.pt | /home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/rdd2020_cls4.yaml

weight_path = "/home/ubuntu/data1/lorenzo/Detection/ultralytics/runs/bt_hw/cls2_rdd_20231016_v0.4_canada/weights/best.pt"
data_path = "/home/ubuntu/data1/lorenzo/Detection/ultralytics/ultralytics/cfg/datasets/bt_hw/cls2_20231005_v0.4_canada.yaml"
imgsz = 320
batch = 32
conf = 0.45
iou = 0.6

model = YOLO(weight_path)
metrics = model.val(data=data_path, 
                    imgsz=imgsz,
                    batch=batch,
                    # save_json=True,
                    # save_txt=True,
                    # save_conf=True,
                    conf=conf,
                    iou=iou)


print(f"map50@95: {metrics.box.map}")
print(f"map50: {metrics.box.map50}")
print(metrics.box.maps)



# data	        None	   path to data file, i.e. coco128.yaml
# imgsz	        640	       size of input images as integer
# batch	        16	       number of images per batch (-1 for AutoBatch)
# save_json	    False	   save results to JSON file
# save_hybrid	False	   save hybrid version of labels (labels + additional predictions)
# conf	        0.001	   object confidence threshold for detection
# iou	        0.6	       intersection over union (IoU) threshold for NMS
# max_det	    300	       maximum number of detections per image
# half	        True	   use half precision (FP16)
# device	    None	   device to run on, i.e. cuda device=0/1/2/3 or device=cpu
# dnn	        False	   use OpenCV DNN for ONNX inference
# plots     	False	   show plots during training
# rect	        False	   rectangular val with each batch collated for minimum padding
# split	        val	       dataset split to use for validation, i.e. 'val', 'test' or 'train'