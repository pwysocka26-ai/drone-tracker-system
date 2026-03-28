import cv2
import numpy as np

print("start")

img = np.zeros((600, 800, 3), dtype=np.uint8)
img[:] = (0, 100, 200)

cv2.imshow("test", img)

print("window should be visible")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("done")
