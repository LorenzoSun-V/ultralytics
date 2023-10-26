from ultralytics import YOLO

model = YOLO('/home/ubuntu/data1/lorenzo/Detection/rknn-yolo/ultralytics_yolov8/runs/detect/train/weights/best.pt')

results = model("/home/ubuntu/data1/datatset/bt/kjg_multi/raw_zips/20230927_1floor", save=True, imgsz=640, conf=0.25)

# args:
#     conf(float): confidence threshold
#     iou(float): nms threshold
#     imgsz(int or tuple): image size
#     half(bool): use fp16
#     device(str): 0/1/2/3/cpu
#     save(bool): save images with results
#     save_txt(bool): save results as .txt file
#     save_conf(bool): save results with confidence scores