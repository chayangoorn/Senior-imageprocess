import cv2
import numpy as np
from skimage.morphology import (erosion, dilation, closing, opening, area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import square

def arcLengths(contours, img):
	i = 0; frames = []; cnts = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 1000:
			peri = cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
			print(len(approx))
			x,y,w,h = cv2.boundingRect(approx)
			frames.append((x,y,w,h))
			cnts.append(approx)
			d2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			i+=1
	if i == 0: return "No area found"
	return (d2, frames, cnts)

def genE(w, h):
	E = np.zeros((h,w))
	E[:,:int(E.shape[1]/2)] = 255
	n = int(E.shape[0]/5)
	for i in range(5):
		if i in [0,2]: E[n*i:n*(i+1),int(E.shape[1]/2):] = 255
		elif i==4: E[n*i:,int(E.shape[1]/2):] = 255
	cv2.imwrite("template.jpg", E)
	return E

def genEback(w, h):
	E = np.zeros((h,w))
	E[:,int(E.shape[1]/2):] = 255
	n = int(E.shape[0]/5)
	for i in range(5):
		if i in [0,2]: E[n*i:n*(i+1),:int(E.shape[1]/2)] = 255
		elif i==4: E[n*i:,:int(E.shape[1]/2)] = 255
	cv2.imwrite("template.jpg", E)
	return E

def matTemp(image, template):
	image = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)[1]
	result = cv2.matchTemplate(image,template,cv2.TM_CCORR_NORMED)
	minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(result)
	loc = maxLoc
	w, h = template.shape[::-1]
	toploc = (loc[0] + w, loc[1] + h)
	d = cv2.rectangle(image,loc,toploc,(0,0,255),1)
	return (d, loc[0],loc[1],w,h,maxVal)

def findBiggest(contours, img):
	if len(contours) != 0:
		d1 = cv2.drawContours(img, contours, -1, 255, 1)
		max_area = 0; biggest = 0
		for i,c in enumerate(contours):
			area = cv2.contourArea(c)
			if area >= 1000:
				if area > max_area: 
					max_area = area
					biggest = i
		if biggest!=0:
			x,y,w,h = cv2.boundingRect(contours[biggest])
    	# draw the biggest contour (c) in green
			d2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			return (d2, x, y, w, h)
		else: return "Not found"
	else: return "Not have contours"

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

def trialK(staff, staff_height, vrt, weight):
	first_h = []
	for j in range(8):
		if j==0:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genE(int(staff.shape[1]/2), int(staff_height)-k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				section = cv2.cvtColor(staff[vrt:vrt+int(staff_height), :int(staff.shape[1]/2)], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp) 
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==1:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genE(int(staff.shape[1]/2), int(staff_height)-k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				temp = cv2.bitwise_not(temp)
				section = cv2.cvtColor(staff[vrt:vrt+int(staff_height), :int(staff.shape[1]/2)], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp) 
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==2:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genEback(int(staff.shape[1]/2), int(staff_height)-k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				section = cv2.cvtColor(staff[vrt:vrt+int(staff_height), :int(staff.shape[1]/2)], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==3:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genEback(int(staff.shape[1]/2), int(staff_height)-k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				temp = cv2.bitwise_not(temp)
				section = cv2.cvtColor(staff[vrt:vrt+int(staff_height), :int(staff.shape[1]/2)], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==4:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genE(int(staff.shape[1]/2), int(staff_height)+k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				section = cv2.cvtColor(staff[vrt:vrt+2*int(staff_height), int(staff.shape[1]/2):], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==5:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genE(int(staff.shape[1]/2), int(staff_height)+k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				temp = cv2.bitwise_not(temp)
				section = cv2.cvtColor(staff[vrt:vrt+2*int(staff_height), int(staff.shape[1]/2):], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==6:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genEback(int(staff.shape[1]/2), int(staff_height)+k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				section = cv2.cvtColor(staff[vrt:vrt+2*int(staff_height), int(staff.shape[1]/2):], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
		elif j==7:
			prob = []; draws = []; height = []; vertical = []
			for k in range(int(staff_height*(weight*0.01))):
				genEback(int(staff.shape[1]/2), int(staff_height)+k)
				temp = cv2.threshold(cv2.imread("template.jpg", 0), 128, 255, cv2.THRESH_OTSU)[1]
				temp = cv2.bitwise_not(temp)
				section = cv2.cvtColor(staff[vrt:vrt+2*int(staff_height), int(staff.shape[1]/2):], cv2.COLOR_BGR2GRAY)
				res = matTemp(section, temp)
				(d, x, y, w, h, p) = res
				prob.append(p); draws.append(d); height.append(h); vertical.append(y)
			if len(prob)>0:
				minus = np.argmax(np.array(prob))
				first_h.append([prob[minus], minus, draws[minus], height[minus], vertical[minus]])
			else: pass
	if len(first_h)>0:
		srt = sorted([(f[0],i) for i,f in enumerate(first_h)], reverse=True)
		#cv2.imwrite("output.jpg", first_h[srt[0][1]][2])
		#cv2.imshow("out",first_h[srt[0][1]][2])
		#cv2.waitKey(0)
		return first_h[srt[0][1]]
	else: return False