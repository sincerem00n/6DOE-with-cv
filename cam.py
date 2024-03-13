import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# classify
def NutScrew(rect):
    w, h = rect[1]
    dx = abs(w-h)
    # print('dx:',dx)
    err = 10
    if dx < err:
        # check nut type
        m4nut = 25
        m6nut = 35
        if w < m4nut+err and w > m4nut-err:
            t = 'nut'
            s = 'm4'
            l = 'None'
            return t, s, l
        elif w < m6nut+err and w > m6nut-err:
            t = 'nut'
            s = 'm6'
            l = 'None'
            return t, s, l
    elif dx > err:
        # check screw type
        m4w = 10
        m6w = 20
        werr = 5
        l18 = 85
        l35 = 145
        lerr = 20

        if w < m4w+werr and w > m4w-werr:
            if h < l18+lerr and h > l18-lerr:
                t = 'screw'
                s = 'm4'
                l = '18'
                return t, s, l
            elif h < l35+lerr and h > l35-lerr:
                t = 'screw'
                s = 'm4'
                l = '35'
                return t, s, l
        elif w < m6w+werr and w > m6w-werr:
            if h < l18+lerr and h > l18-lerr:
                t = 'screw'
                s = 'm6'
                l = '18'
                return t, s, l
            elif h < l35+lerr and h > l35-lerr:
                t = 'screw'
                s = 'm6'
                l = '35'
                return t, s, l


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

    img_resize = gray[215:780, 650:1450]

    _, img_tres = cv2.threshold(img_resize, 135, 255, cv2.THRESH_TOZERO)

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
    contours = [i for i in contours if cv2.contourArea(i) >= 400]
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

    img2 = cv2.merge([img_resize, img_resize, img_resize])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if rect[1][0] > rect[1][1]:
            rect = (rect[0], (rect[1][1], rect[1][0]), rect[2])
        result = NutScrew(rect)
        if result is not None:
            t, s, l = result
            text = f'{t} {s} {l}'
        else:
            text = 'None'
        cv2.putText(img2, text, (x, y+h), font, 0.7, (0,255,0), 2, cv2.LINE_AA)

    # plt.imshow(img2, cmap='gray')

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('frame', img2)
    # time.sleep(1)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Todo
# 1. fixed camera
# 2. fix contour counting
# 3. adjust screw and nut size