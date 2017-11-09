import cv2
import numpy as np

cap = cv2.VideoCapture('../data/trim_start_end.mp4')
# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# out = cv2.VideoWriter('../results/output_MOG2.m4v',fourcc, 20.0, (480,640), False)
# fgbg = cv2.createBackgroundSubtractorMOG2(5, 15, True) #history, varThreshold, bShadowDetection
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if i == 100:
        background = gray
        cv2.imwrite('../results/ours_image_{}.jpg'.format(i), background)
        break

while True:
    i += 1
    ret, frame = cap.read()
    height, width, layers = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray - background
    # for i in range(height):
    #     for j in range(width):
    #             if gray[i][j] - background[i][j] < 20:
    #                 gray[i][j] = 0
    #             else: gray[i][j] = 255

    # fgmask = fgbg.apply(frame)
    #gray = cv2.fastNlMeansDenoisingMulti(frame, 2, 5, None, 4, 7, 35)
    cv2.imshow('frame', gray)
    if i == 200:
        cv2.imwrite('../results/ours_image_{}.jpg'.format(i), gray)
    #out.write(fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyALLWindows()