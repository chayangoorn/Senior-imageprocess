import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray


img = imread("staff3.png")
gray = rgb2gray(img)
imshow(img)
imshow(gray, cmap="gray")
plt.show()