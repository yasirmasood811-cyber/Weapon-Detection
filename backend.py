from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import cv2

app = FastAPI()

model = YOLO("weapon_best.pt")

@app.get("/")
def welcome_msg():
    return {"welcome": "YOLO model is ready for weapon detection!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model.predict(source=temp_file)
        annotated = results[0].plot()

        cv2.imwrite("output.jpg", annotated)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class": model.names[cls],
                    "confidence": conf,
                    "bbox": xyxy
                })

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
