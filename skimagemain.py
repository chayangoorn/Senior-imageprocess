import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import genE, matTemp
import cv2

temp = cv2.imread("E-template.jpg", 0)
img = cv2.imread("staff1.jpg")
(d, x, y, w, h) = matTemp(img, temp)
cv2.imshow("output", d)
cv2.waitKey(0)