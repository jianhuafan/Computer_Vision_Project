import numpy as np 
import cv2

cap = cv2.VideoCapture('WeChatSight418.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
i = 0
while True:
	ret, frame = cap.read()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame', fgmask)
	i += 1
	# if i == 700 or 800 or 900:
	# 	cv2.waitKey(0)
	out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()

