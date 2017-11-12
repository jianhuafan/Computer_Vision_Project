import cv2
import numpy as np

cap = cv2.VideoCapture('../results/output_CandyEdge.m4v')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/EDGE_MOG2.m4v',fourcc, 20.0, (480,640), False)
fgbg = cv2.createBackgroundSubtractorMOG2(10, 25, True) #history, varThreshold, bShadowDetection
i = 0
while True:
	i += 1
	ret, frame = cap.read()
	height, width, layers = frame.shape
	fgmask = fgbg.apply(frame)
	blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
	cv2.imshow('frame', blur)
	out.write(fgmask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()