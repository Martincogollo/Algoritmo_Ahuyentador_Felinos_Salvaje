import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('./PruebaModelo/model/best_felinos_50.pt')

while True:
    ret, frame = cap.read()

    results = model.predict(frame, imgsz = 640, conf = 0.2)
    if len(results) != 0:
        for res in results:
            print("Felino detectado")

        annotated_frames = results[0].plot()

    cv2.imshow('Felino detectado', annotated_frames)
    t=cv2.waitKey(10)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()