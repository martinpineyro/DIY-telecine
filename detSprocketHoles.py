import cv2
import numpy as np

img = cv2.imread('8mm.jpg',0)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 255


# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 100000


detector = cv2.SimpleBlobDetector(params)


# Detect blobs.
keypoints = detector.detect(th2)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(th2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

   
