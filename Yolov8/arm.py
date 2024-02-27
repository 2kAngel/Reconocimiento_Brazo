
from ultralytics import YOLO
import cv2
import os
model = YOLO("EntrenamientoBrazo/best.pt")#Ruta relativa no disponible, usar absoluta a best.pt  , lo mismo en data.yaml
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    resultados = model.predict(frame, imgsz=640, conf=0.25)
    anotaciones = resultados[0].plot()
    cv2.imshow("Deteccion y segmentacion", anotaciones)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()