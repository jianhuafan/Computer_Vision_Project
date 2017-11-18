import cv2
import numpy as np

cap = cv2.VideoCapture('../results/output_MOG2.m4v')
i = 0
while True:
	i += 1
	ret, frame = cap.read()
	#height, width, layers = frame.shape
	if i == 800:
		cv2.imwrite('../results/MOG2_original_{}.png'.format(i), frame)
	open_kernel = np.ones((5, 1), np.uint8)
	close_kernel = np.ones((30, 30), np.uint8)
	opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, open_kernel)
	dilation = cv2.dilate(frame, open_kernel)
	if i == 800:
		cv2.imwrite('../results/MOG2_dilation_{}.png'.format(i), dilation)
	closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, close_kernel)
	if i == 800:
		cv2.imwrite('../results/MOG2_closing_{}.png'.format(i), closing)
	cv2.imshow('frame', dilation)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyALLWindows()