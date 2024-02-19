import cv2
import matplotlib.pyplot as plt

rin = cv2.imread('staff5.jpg')
plt.figure(figsize=[6,7])
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
for i,c in enumerate('bgr'):
    hist = cv2.calcHist([rin],[i],None,[256],[0,256])
    ax1.plot(hist,c) # ฮิสโทแกรม
    ax2.plot(hist.cumsum(),c) # ผลบวกสะสม
plt.tight_layout()
plt.show()