import cv2
import numpy as np
from skimage.morphology import (erosion, dilation, closing, opening, area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import square

def findBiggest(contours, img):
    if len(contours) != 0:
    # draw in blue the contours that were founded
        d1 = cv2.drawContours(img, contours, -1, 255, 1)

    # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        print(cv2.contourArea(c))
        x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
        d2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return d2, x, y, w, h

def findShortestHeight(contours, img):
	if len(contours) != 0:
    # draw in blue the contours that were founded
		d1 = cv2.drawContours(img, contours, -1, 255, 1)
		rects = []
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			rects.append((h,w,x,y))
        #c = min(contours, key = cv2.contourArea)
        #print(cv2.contourArea(c))
		rects = sorted(rects)
		print(rects)
		d2 = cv2.rectangle(img,(rects[0][2],rects[0][3]),(rects[0][2]+rects[0][1],rects[0][3]+rects[0][0]),(0,255,0),2)
	#return d2, x, y, w, h
	return d2, rects[0]

def findLongestHeight(contours, img):
	if len(contours) != 0:
    # draw in blue the contours that were founded
		d1 = cv2.drawContours(img, contours, -1, 255, 1)
		rects = []
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			rects.append((h,w,x,y))
        #c = min(contours, key = cv2.contourArea)
        #print(cv2.contourArea(c))
		rects = sorted(rects, reverse=True)
		print(rects)
		d2 = cv2.rectangle(img,(rects[0][2],rects[0][3]),(rects[0][2]+rects[0][1],rects[0][3]+rects[0][0]),(0,255,0),2)
	#return d2, x, y, w, h
	return d2, rects[0]

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def sobel_edge_detector(img):
	grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
	grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
	grad = np.sqrt(grad_x**2 + grad_y**2)
	grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
	return grad_norm

def multi_dil(im, num, element=square(3)):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element=square(3)):
    for i in range(num):
        im = erosion(im, element)
    return im