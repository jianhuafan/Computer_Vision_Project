import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('../data/input.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/output_CannyEdge.m4v',fourcc, 20.0, (544,960), False)
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    sigma = 0.33
    v = np.median(frame)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)
    edges = cv2.Canny(blur, lower, upper)
    cv2.imshow('frame', edges)
    out.write(edges)
    #height, width, layers = frame.shape
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyALLWindows()