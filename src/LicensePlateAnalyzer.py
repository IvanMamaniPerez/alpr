import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import re


class LicensePlateAnalyzer:
    
    def __init__(
            self, 
            model_path = "runs/detect/train/weights/best.pt", 
            confidence = 70, 
            pattern = r"\b[a-zA-Z][a-zA-Z\d][a-zA-Z]-\d{3}\b"
        ):
        
        self.model = YOLO(model_path)
        self.pattern = pattern
        self.confidence = confidence
    
    def predict(self, frame):
        list_pred = self.model.predict(frame)
        pred = list_pred[0]          
        boxes = pred.boxes.xyxy.tolist()
        results = []
        if(len(boxes) > 0): 
            box = boxes[0]
            image_croped = self.crop_box(box, frame)
            results = self.extract_license_plate(image_croped)
        return results

    def crop_box(self, box, frame):
        x1, y1, x2, y2 = box    
        return frame[int(y1):int(y2), int(x1):int(x2)]                    
                            
    def extract_license_plate(self, image_croped):
        data = pytesseract.image_to_data(image_croped, config='--psm 6',output_type=pytesseract.Output.DICT)
        results = []
        for i in range(len(data['text'])):
            if len(data['text'][i]) >= 7  and  int(data['conf'][i]) > 70:
                text = data['text'][i]
                confidence = data['conf'][i]
                matches = re.findall(self.pattern, text)
                if len(matches) > 0:
                    matches = matches[0]
                    matches = matches.upper()
                    results.append({'license':matches, 'confidence':confidence, 'text':text})
                    
        return results
    
    