from ultralytics import YOLO 
 

if __name__ == "__main__":
    
    model = YOLO('yolov8n.pt')

    model.train(data="train.yaml", epochs=5, imgsz=(472, 303), batch=4, optimizer="Adam", device = 0)

