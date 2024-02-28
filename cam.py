import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

screw_w = 40
nut_w = 60
err = 10

def classify(rect):
    w, h = rect[1]
    if w > screw_w and w < screw_w+err and h > screw_w :
        return 'screw'
    elif w > nut_w and w < nut_w+err:
        return 'nut'
    else:
        return 'unknown'


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_resize = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    # print(img_resize.shape)
    # plt.imshow(img_resize, cmap='gray') 

    _, img_tres = cv2.threshold(img_resize, 135, 255, cv2.THRESH_TOZERO_INV)

    # closing
    ksize = 7
    kernel = np.ones((ksize, ksize), np.uint8)
    closing = cv2.morphologyEx(img_tres,
                            cv2.MORPH_CLOSE,
                            kernel,
                            iterations=1)
    
    _,bw = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.merge([bw, bw, bw])
    
    # draw contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i])
        print(f'area{i}:',a)
        cv2.drawContours(mask, contours, i, (255, 0, 0), 2)

        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, str(i), (x, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # min area rect
        rect = cv2.minAreaRect(contours[i]) # (cen(x, y), (w, h), angle)
        # angle -> gripper head angle
        print(f'rect{i}:',rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(mask, [box], 0, (0, 255, 255), 2)

    # classify
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        print(f'obj{i}:', classify(rect))

    img2 = cv2.merge([img_resize, img_resize, img_resize])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.putText(img2, classify(rect), (x, y+h), font, 1, (0,255,0), 2, cv2.LINE_AA)

    # plt.imshow(img2, cmap='gray')

    # Display the resulting frame
    cv2.imshow('mask', mask)
    cv2.imshow('frame', img2)
    time.sleep(1)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Todo
# 1. fixed camera
# 2. fix contour counting
# 3. adjust screw and nut size