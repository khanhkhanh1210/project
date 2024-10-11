from ultralytics import YOLO

def train():
    # Load YOLOv8 model
    model =  YOLO('yolov8n.pt')
    # Load YOLOv8 model using cuda
    model.train(
        data='license_plate.yaml',
        imgsz=640, 
        epochs=20, 
        device=0,
        batch=16,        
        project='license_plate',
        amp=False,
    )

if __name__ == '__main__':
    train()