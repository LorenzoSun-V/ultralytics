'''
Author: 孙家辉 sunjiahui@boton-tech.com
Date: 2023-11-08 02:02:29
LastEditors: 孙家辉 sunjiahui@boton-tech.com
LastEditTime: 2023-11-08 02:51:37
Description: 
'''
import sys
sys.path.append('./')
from ultralytics import YOLO

model = YOLO("yolov8s.yaml")
model = YOLO("yolov8s.pt")

# hw
model.train(data="ultralytics/cfg/datasets/hwir/cls2_20231107_1floor.yaml", 
            epochs=50, 
            imgsz=320,
            batch=64,
            device=0,
            project="runs/hwir",
            name="cls2_20231107_1floor",
            exist_ok=True,
            patience=20,
)

# model	            None	    path to model file, i.e. yolov8n.pt, yolov8n.yaml
# data	            None	    path to data file, i.e. coco128.yaml
# epochs	        100	        number of epochs to train for
# patience	        50	        epochs to wait for no observable improvement for early stopping of training
# batch	            16	        number of images per batch (-1 for AutoBatch)
# imgsz	            640	        size of input images as integer
# save	            True	    save train checkpoints and predict results
# save_period	    -1	        Save checkpoint every x epochs (disabled if < 1)
# cache	            False	    True/ram, disk or False. Use cache for data loading
# device	        None	    device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
# workers	        8	        number of worker threads for data loading (per RANK if DDP)
# project	        None	    project name
# name	            None	    experiment name
# exist_ok	        False	    whether to overwrite existing experiment
# pretrained	    True	    (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)
# optimizer	        'auto'	    optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
# verbose	        False	    whether to print verbose output
# seed	            0	        random seed for reproducibility
# deterministic	    True	    whether to enable deterministic mode
# single_cls	    False	    train multi-class data as single-class
# rect	            False	    rectangular training with each batch collated for minimum padding
# cos_lr	        False	    use cosine learning rate scheduler
# close_mosaic	    10	        (int) disable mosaic augmentation for final epochs (0 to disable)
# resume	        False	    resume training from last checkpoint
# amp	            True	    Automatic Mixed Precision (AMP) training, choices=[True, False]
# fraction	        1.0	        dataset fraction to train on (default is 1.0, all images in train set)
# profile	        False	    profile ONNX and TensorRT speeds during training for loggers
# freeze	        None	    (int or list, optional) freeze first n layers, or freeze list of layer indices during training
# lr0	            0.01	    initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
# lrf	            0.01	    final learning rate (lr0 * lrf)
# momentum	        0.937	    SGD momentum/Adam beta1
# weight_decay	    0.0005	    optimizer weight decay 5e-4
# warmup_epochs	    3.0	        warmup epochs (fractions ok)
# warmup_momentum	0.8	        warmup initial momentum
# warmup_bias_lr	0.1	        warmup initial bias lr
# box	            7.5	        box loss gain
# cls	            0.5	        cls loss gain (scale with pixels)
# dfl	            1.5	        dfl loss gain
# pose	            12.0	    pose loss gain (pose-only)
# kobj	            2.0	        keypoint obj loss gain (pose-only)
# label_smoothing	0.0	        label smoothing (fraction)
# nbs	            64	        nominal batch size
# overlap_mask	    True	    masks should overlap during training (segment train only)
# mask_ratio	    4	        mask downsample ratio (segment train only)
# dropout	        0.0	        use dropout regularization (classify train only)
# val	            True	    validate/test during training