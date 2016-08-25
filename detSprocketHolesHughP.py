import cv2
import numpy as np

img = cv2.imread('8mm.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#PENDIENTE: Probar definiendo una ROI en la banda superior del film (donde van a estar los agujeros)

edges = cv2.Canny(gray,20,200,apertureSize = 3)


minLineLength = 60
maxLineGap = 1

'''
First parameter, Input image should be a binary image, so apply threshold or use canny edge detection before finding applying hough transform. 
Second and third parameters are rho and theta accuracies respectively. 
Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line. 
Fifth parameter is Maximum allowed gap between line segments to treat them as single line.
'''
lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)

for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('houghlines',img)
cv2.waitKey(0)
