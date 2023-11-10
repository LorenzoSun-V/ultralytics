'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-08 01:58:36
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2023-11-08 03:20:29
Description: 
'''
import sys
sys.path.append('./')
from ultralytics import YOLO

model = YOLO('/lorenzo/bt_repo/ultralytics/runs/hwir/cls2_20231107_1floor/weights/best.pt')

results = model("/data/bt/hw_multi/raw_zips/ir/buliangou/20231101/slide/", 
                save=True, 
                save_txt=True,
                save_conf=True,
                project="runs/hwir",
                name="cls2_20231107_1floor_detect",
                imgsz=320, 
                conf=0.25,
                iou=0.65)

# args:
#     conf(float): confidence threshold
#     iou(float): nms threshold
#     imgsz(int or tuple): image size
#     half(bool): use fp16
#     device(str): 0/1/2/3/cpu
#     save(bool): save images with results
#     save_txt(bool): save results as .txt file
#     save_conf(bool): save results with confidence scores