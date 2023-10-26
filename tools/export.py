import sys
sys.path.append('./')
from ultralytics import YOLO

model = YOLO('/home/ubuntu/data1/lorenzo/Detection/ultralytics/bt_hw/cls2_20231005_v0.4_canada/weights/best.pt')

model.export(format='onnx', simplify=True)


# format	'torchscript'	format to export to
# imgsz	    640	            image size as scalar or (h, w) list, i.e. (640, 480)
# keras	    False	        use Keras for TF SavedModel export
# optimize	False	        TorchScript: optimize for mobile
# half	    False	        FP16 quantization
# int8	    False	        INT8 quantization
# dynamic	False	        ONNX/TensorRT: dynamic axes
# simplify	False	        ONNX/TensorRT: simplify model
# opset	    None	        ONNX: opset version (optional, defaults to latest)
# workspace	4	            TensorRT: workspace size (GB)
# nms	    False	