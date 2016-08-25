import cv2
import numpy as np

img = cv2.imread('8mm.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#PENDIENTE: Probar definiendo una ROI en la banda superior del film (donde van a estar los agujeros)

edges = cv2.Canny(gray,50,200,apertureSize = 3)


'''
First parameter, Input image should be a binary image, so apply threshold or use canny edge detection before finding applying hough transform. 
Second and third parameters are rho and theta accuracies respectively. 
Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line. 
'''
lines = cv2.HoughLines(edges,1,np.pi/180,50)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('houghlines',img)
cv2.waitKey(0)
