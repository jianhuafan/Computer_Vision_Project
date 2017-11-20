import cv2
import numpy as np 

im = cv2.imread('../results/image/MOG2_557.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
# blur = cv2.GaussianBlur(im, (5, 5), 0)


# open_kernel = np.ones((1, 2), np.uint8)
# opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, open_kernel)
# scale = cv2.resize(im, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

im_copy = im.copy()

h, w = im.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(im_copy, mask, (h/2,w/2), 255)

im_copy_inv = cv2.bitwise_not(im_copy)

output = im | im_copy_inv
# close_kernel = np.ones((10, 10), np.uint8)

cv2.imshow('original', im)
cv2.imshow('Frame', output)
#cv2.imwrite('../results/image/output1.png', output)
cv2.waitKey(0)