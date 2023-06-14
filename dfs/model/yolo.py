from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/beegfs/home/robert.jarolim/projects/yolov5/gregor.yaml", epochs=100, imgsz=1024, batch=4)  # train the model
success = model.export()