from ultralytics import YOLO

# 加载模型
model = YOLO(model="airplane.pt")

results = model(source="aircraft.jpg", save=True)