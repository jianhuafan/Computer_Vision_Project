import numpy as np 
import cv2

cap = cv2.VideoCapture('../data/WeChatSight420.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/output_GMG.m4v',fourcc, 20.0, (480,640), False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(2,0.5)
i = 0
while True:
	i += 1
	ret, frame = cap.read()
	height, width, layers = frame.shape
	fgmask = fgbg.apply(frame)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	if i == 800:
		cv2.imwrite('../results/GMG_image_{}.jpg'.format(i), fgmask)
	cv2.imshow('frame', fgmask)
	out.write(fgmask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()

