import cv2
import numpy as np
import time
import PoseModule as pm

#cap = cv2.VideoCapture('C:/Users/Lorena/Desktop/Ángel/TFG/Yolov8/EntrenamientoBrazo/videoBrazoAngel.mp4')
cap = cv2.VideoCapture('0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = pm.poseDetector()
max_movement_percentage = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Calculating the angle of the arm
        angle = detector.findAngle(img, 12, 14, 16)
        
        # Interpolating the angle to get the movement percentage
        movement_percentage = np.interp(angle, (210, 310), (0, 100))
        
        # Storing the maximum movement percentage
        max_movement_percentage = max(max_movement_percentage, movement_percentage)
        
        # Displaying the movement percentage
        cv2.putText(img, f'Max Movement: {int(max_movement_percentage)}%', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
