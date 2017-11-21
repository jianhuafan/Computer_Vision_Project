import numpy as np
import cv2

gray = cv2.imread('11.png')
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
minLineLength=3
new = np.zeros(gray.shape[:2], np.uint8)
lines = cv2.HoughLinesP(image=gray,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=200)

a,b,c = lines.shape
for i in range(a):
    x = lines[i][0][0] - lines [i][0][2]
    y = lines[i][0][1] - lines [i][0][3]
    if x!= 0:
        if abs(y/x) <1:
            cv2.line(new, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 1)
temp = cv2.bitwise_and(new, gray)
gray = cv2.subtract(gray,temp)
cv2.imwrite('houghlines.jpg', new)
cv2.imshow('img', gray)
cv2.waitKey(0)