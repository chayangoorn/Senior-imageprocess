import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("E.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bi = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU)[1]
cv2.imwrite("E-template.jpg", bi)

cv2.imshow("Output", bi)
cv2.waitKey(0)

