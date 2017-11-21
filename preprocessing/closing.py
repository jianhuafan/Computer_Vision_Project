import cv2
import numpy as np

cap = cv2.VideoCapture('../results/video/CannyEdge_MOG2.m4v')
i = 0
element = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/video/closing.m4v',fourcc, 20.0, (544,960), False)
while True:
	i += 1
	ret, frame = cap.read()
	#mask = cv2.erode(frame, element)
	# mask = cv2.dilate(frame, element)
	#mask = cv2.erode(mask, element)
	#height, width, layers = frame.shape
	# # open_kernel = np.ones((4, 1), np.uint8)
	fgmask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# _, cnts, _= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# for c in cnts:
	# 	area = cv2.contourArea(c)
	# 	# if area < 900:
	# 	# 	cv2.drawContours(fgmask, [c], -1, 0, -1)
	# 	# else:
	# 	cv2.fillPoly(frame, c, color=(255,255,255))
	# closing = frame
	# close_kernel = np.ones((10, 10), np.uint8)
	opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)
	# # dilation = cv2.dilate(frame, open_kernel)
	# # if i == 800:
	# # 	cv2.imwrite('../results/MOG2_dilation_{}.png'.format(i), dilation)
	# for i in range(10):
	# 	closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, close_kernel)
	# # if i == 800:
	# 	cv2.imwrite('../results/MOG2_closing_{}.png'.format(i), closing)
	cv2.imshow('frame', opening)
	out.write(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyALLWindows()