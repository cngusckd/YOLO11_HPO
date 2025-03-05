from ultralytics import YOLO

model = YOLO("yolov11n.yaml")

result = model.train(data = 'voc.yaml', epochs = 20, imgsz = 640)