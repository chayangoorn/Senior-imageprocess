import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import adjust_gamma, sobel_edge_detector, findBiggest, findShortestHeight, multi_dil, multi_ero, findLongestHeight
from skimage.morphology import (erosion, dilation, closing, opening, area_closing)
import math
from skimage.morphology import square
from rembg import remove
#import easyocr

img = cv2.imread("staff4.jpg")
img1 = cv2.resize(img, (int((480/img.shape[0])*img.shape[1]), 480))
r = remove(img1)
img2 = img1.copy()
gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
gamma = adjust_gamma(blur, 0.5)
#thresh,binarize = cv2.threshold(subgamma, 128, 255, cv2.THRESH_BINARY)
bi = cv2.threshold(gamma, 200, 255, cv2.THRESH_OTSU)[1]
opp = opening(bi, square(5))
sobel = sobel_edge_detector(bi)
contours, hierarchy = cv2.findContours(sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

draws,x,y,w,h = findBiggest(contours, img1)
img2 = img2[y:y+h, x:x+w]
img3 = img2.copy()
img4 = img3.copy()

gray1 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
blur1 = cv2.medianBlur(gray1, 3)
gamma1 = adjust_gamma(blur1, 0.5)
op = opening(gamma1, square(5))
dii = dilation(op, square(5))
bi1 = cv2.threshold(gamma1, 128, 255, cv2.THRESH_OTSU)[1]
"""curImg = bi1.copy()
template = cv2.imread("E-template.jpg")
height,width,_ = template.shape
template = cv2.threshold(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)[1]
templateMap = cv2.matchTemplate(curImg, template, cv2.TM_CCOEFF_NORMED)
_,_,minLoc,maxLoc = cv2.minMaxLoc(templateMap)
topleft = maxLoc
bottomright = (topleft[0]+width, topleft[1]+height)
rec = cv2.rectangle(curImg, topleft, bottomright, (0,255,0), 2)
"""
op1 = opening(bi1, square(5))
dl = dilation(op1, square(3))
close = closing(op1, square(5))
dil = dilation(close, square(3))
sobel1 = sobel_edge_detector(op1)

contours1, hierarchy1 = cv2.findContours(sobel1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
d1,win_size = findLongestHeight(contours1, img3)
h = win_size[0]

num = math.floor(img4.shape[0]/h)
imgsections = []
for i in range(1,num+1):
    imgsections.append(img4[img4.shape[0]-(h*i):img4.shape[0]-(h*(i-1)),])

shows = np.concatenate(tuple([u for u in imgsections]), axis=1)

cv2.imshow("output",shows)
#cv2.imshow("output",label_im.astype(np.uint8))
cv2.waitKey(0)



