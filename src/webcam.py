import numpy as np
import cv2
from ultralytics import YOLO
import pytesseract
import re


if __name__ == "__main__":

    model = YOLO("runs/detect/train/weights/best.pt") 
    
    patronPeruvianLicensePlate = r"\b[a-zA-Z][a-zA-Z\d][a-zA-Z]-\d{3}\b"
    
    cap = cv2.VideoCapture(0)
    
    arrayFindeds = []
    
    while(True):
        
        ret, frame = cap.read()
        
        listPred = model.predict(frame)

        pred = listPred[0]
        
        if pred is None :
            continue
        
        frame = pred.plot()
        
        #extract box from prediction
        boxes = pred.boxes.xyxy.tolist()
            
        for i, box in enumerate(boxes):
        
            x1, y1, x2, y2 = box
            
            ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
            
            cv2.imshow('procesed',ultralytics_crop_object)
        
            data = pytesseract.image_to_data(ultralytics_crop_object, config='--psm 6',output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if len(data['text'][i]) >= 7  and  int(data['conf'][i]) > 70:  # Filtrar valores de confianza inválidos
                    print(f"Texto: {data['text'][i]} - Confianza: {data['conf'][i]}%")
            
                    text = data['text'][i]
                    
                    print(text)
                    
                    finalText = re.findall(patronPeruvianLicensePlate, text)
                    
                    if len(finalText) > 0:
                        finalText = finalText[0];
                        
                        #conver to uppercase
                        finalText = finalText.upper()
                        
                        if finalText not in arrayFindeds:
                            arrayFindeds.append(finalText) 
                            
                            print(finalText)
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
    print(arrayFindeds)