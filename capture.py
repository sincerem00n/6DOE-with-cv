import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

cam = cv2.VideoCapture(0)
n = 10
path = 'src/'
while True:
    for i in range(n):
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            break
        cv2.imshow('frame', frame)
        cv2.imwrite(f"{path}image_{i}.jpg", frame)
        time.sleep(1)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()