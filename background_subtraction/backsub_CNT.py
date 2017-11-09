import numpy as np 
import cv2

cap = cv2.VideoCapture('../data/WeChatSight420.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/output_CNT.m4v',fourcc, 20.0, (480,640), False)
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(15, True, 15*60, True)
#minStability, useHistory, maxStability, isParallel
i = 0
while True:
	i += 1
	ret, frame = cap.read()
	height, width, layers = frame.shape
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame', fgmask)
	if i == 800:
		cv2.imwrite('../results/CNT_image_{}.jpg'.format(i), fgmask)
	out.write(fgmask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()

