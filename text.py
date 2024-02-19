import cv2
import numpy as np
from utils import adjust_gamma, sobel_edge_detector, findBiggest, multi_dil, multi_ero, findLongestHeight, arcLength
from skimage.morphology import (erosion, dilation, closing, opening, area_closing)
from skimage.morphology import square

img = cv2.imread("staff9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gamma = adjust_gamma(gray, 2)
op = opening(gamma, square(5))
dil = dilation(op, square(5))

sub = cv2.subtract(gamma, dil)
cv2.imshow("output", sub)
cv2.waitKey(0)