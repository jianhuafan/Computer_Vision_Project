import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('../data/WeChatSight420.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('../results/output_CandyEdge.m4v',fourcc, 20.0, (480,640), False)
i = 0
while True:
    i += 1
    ret, frame = cap.read()
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('frame', edges)
    out.write(edges)
    #height, width, layers = frame.shape
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyALLWindows()