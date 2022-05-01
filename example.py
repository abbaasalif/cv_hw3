import cv2
import numpy as np

img = np.zeros((768, 1024, 3), dtype='uint8')

points = np.array([[910, 641], [206, 632], [696, 488], [458, 485]])

cv2.polylines(img, [points], 1, (255,255,255))

winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey()
cv2.destroyWindow(winname)
print(points)