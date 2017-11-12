import cv2
import numpy as np

cap = cv2.VideoCapture('../data/WeChatSight420.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/output_MOG2_clip.m4v',fourcc, 20.0, (480,640), False)
fgbg = cv2.createBackgroundSubtractorMOG2(5, 15, True) #history, varThreshold, bShadowDetection
i = 0
while True:
	i += 1
	ret, frame = cap.read()
	if i == 800:
		cv2.imwrite('../results/original_{}.png'.format(i), frame)
	height, width, layers = frame.shape
	fgmask = fgbg.apply(frame)

	cv2.imshow('frame', fgmask)
	out.write(fgmask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()