
from fastapi import FastAPI, WebSocket, File, UploadFile, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import base64
import json
import numpy as np

# para importar el analizador de placas de la carpeta src
import sys
sys.path.append("src")
from LicensePlateAnalyzer import LicensePlateAnalyzer
analyzer = LicensePlateAnalyzer(model_path="runs/detect/train/weights/best.pt")

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            # Procesar la imagen recibida
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            results = analyzer.predict(img)
            
            text = ""
            confidence = 0
            if len(results) > 0:
                text = results[0]['license']
                confidence = results[0]['confidence']
            # Aquí incluirías el procesamiento específico (e.g., detección de objetos)
            # Simulamos un proceso que etiqueta la imagen y crea un resultado JSON
            processed_img = cv2.putText(img, 'Procesado: ' + text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode('.jpg', processed_img)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            response_data = {
                "message": "Imagen procesada correctamente",
                "license": text,
                "confidence": confidence,
                "img_data": jpg_as_text
            }
            await websocket.send_json(response_data)
    except WebSocketDisconnect:
        print("Client disconnected")


