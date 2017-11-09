import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('../results/MOG2_original_800.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(img,(6,6))
cv2.imwrite('../results/mean_filter.png', blur)
plt.imshow(blur),plt.colorbar(),plt.show()