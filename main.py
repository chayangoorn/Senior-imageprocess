import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import adjust_gamma, sobel_edge_detector, findBiggest, arcLengths, genE, genEback, matTemp, trialK
from skimage.morphology import (erosion, dilation, closing, opening, area_closing)
from skimage.morphology import square
from tqdm import tqdm
from scipy import stats

def get_combined_image_array(picture): # ใช้ matrix จาก grayscale
    color_array = []
    for i in range(len(picture)):
        value = 0
        for j in picture[i]:
            value += j
        value = value/len(picture[i])
        color_array.append(value)
    return color_array

def plot_graph(array):
    x = np.array(array)
    y = np.array([i+1 for i in range(len(array))])
    y_reverse = y[::-1]
    plt.plot(x, y_reverse)
    plt.show()
    return None

def draw_range_line(img,line_dist): #ต้อง copy ก่อน
    for i in range(img.shape[0]//line_dist):
        #start = (0,(i+1)*line_dist)
        start = (0,i*line_dist+img.shape[0]%line_dist)
        #end = (img.shape[1],(i+1)*line_dist)
        end = (img.shape[1],i*line_dist+img.shape[0]%line_dist)
        color = (0, 255, 0) 
        img = cv2.line(img, start, end, color, 1)
    picture = img
    return picture

def draw_steepest_slope(img,array):
    diff_array = abs(np.diff(array))
    point = np.argmax(diff_array)
    start = (0,point)
    end = (img.shape[1],point)
    color = (0, 255, 0) 
    img = cv2.line(img, start, end, color, 1)
    picture = img
    return picture

cropped = []; staff_guage_idx = 0; staff_height = 0
# pre-processing image to crop staff gauge
img = cv2.imread("staff5.jpg")
img1 = cv2.resize(img, (int((480/img.shape[0])*img.shape[1]), 480))
img2 = img1.copy()
gamma = adjust_gamma(img1, 2)

# change color mode to HSV
hsv = cv2.cvtColor(gamma, cv2.COLOR_BGR2HSV)

# Input color range & operation
h1, s1, v1 = (20, 80, 80) #str(input("Lower Hue, Saturation, Value: ")).split()
h2, s2, v2 = (30, 255, 255) #str(input("Upper Hue, Saturation, Value: ")).split()
lower_yellow = np.array([int(h1),int(s1),int(v1)])
upper_yellow = np.array([int(h2),int(s2),int(v2)])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# mask enhancement
dil = dilation(mask, square(5))
er = erosion(dil, square(5))
op = opening(er, square(5))
clos = closing(op, square(5))
cv2.imshow("output",clos)
cv2.waitKey(0)

# find staff guages area to crop it
cnts2,_ = cv2.findContours(clos,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
if len(cnts2)>0:
    res = arcLengths(cnts2, img1)
    if res!="No area found":
        for i in range(len(res[1])):
            img4 = img2.copy()
            img4 = img4[res[1][i][1]:res[1][i][1]+res[1][i][3],res[1][i][0]:res[1][i][0]+res[1][i][2]]
            cropped.append((img4, res[1][i], res[2][i]))
            #draws,x,y,w,h = findBiggest(cnts2, img1)
    else:
        print(res)
else:
    print("No contours found")

# verify cropped image is staff guage
for i, (c, point, ct) in enumerate(cropped):
    cv2.imshow("preview",c)
    cv2.waitKey(0)

    # cropped image enhancement
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (0,0), fx=2, fy=2)
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(s, -1, kernel)
    bi = cv2.threshold(sharpened, 128, 255, cv2.THRESH_OTSU)[1]
    bi = cv2.bitwise_not(bi) 
    r1 = closing(opening(bi, square(3)), square(3))
    r2 = closing(opening(r1, square(3)), square(3))
    r3 = closing(r2, square(3))

    # cropped image find E template (biggest area)
    # if this image is staff guage, the biggest area maybe the E (left/right) 
    sobel = cv2.Canny(bi, 128, 255)
    cnt,_ = cv2.findContours(sobel,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img3 = cv2.resize(c.copy(), (0,0), fx=2, fy=2)
    biggest = findBiggest(cnt, img3)
    if type(biggest) == tuple:
        (d2,x,y,w,h) = biggest

        # matching template by generate E template from half of cropped image width and biggest area height
        t = genE(int(img3.shape[1]/2), h)
        temp = cv2.imread("template.jpg")
        temp = cv2.threshold(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_OTSU)[1]
        temp_mask = cv2.bitwise_not(temp) 
        img4 = r3
        img5 = cv2.resize(c.copy(), (0,0), fx=2, fy=2)
        result = cv2.matchTemplate(img4,temp,cv2.TM_CCORR_NORMED)
        wi, hi = temp.shape[::-1]
        print("width - height:", wi, hi)
        loc = np.where(result >= 0.80)
        for loc1 in zip(*loc[::-1]):
            cv2.rectangle(img5,loc1,(loc1[0] + w, loc1[1] + h),(0,0,255),1)
        print("count matching:", len(loc[0]))
        if len(loc[0]) > 0: 
            staff_guage_idx = i
            staff_height = h
        cv2.imwrite("drawings.jpg", img5)
        cv2.imshow("verify E",img5)
        cv2.waitKey(0)
    else:
        print("Not found area")

# water level reading
staff = cropped[staff_guage_idx][0]
print(staff.shape)
num_staff_frame = staff.shape[0]//staff_height
last_frame = staff.shape[0]%staff_height
for j in range(int(num_staff_frame)+1):
    cv2.line(staff, (0, int(j*staff_height)), (staff.shape[1], int(j*staff_height)), (0,255,0), 2)
cv2.imshow("lines",clos)
cv2.waitKey(0)
print(num_staff_frame, last_frame)

staff = cv2.resize(cropped[staff_guage_idx][0], (0,0), fx=2, fy=2)
height_dataset = []
for z in range(int(staff.shape[0]//staff_height)):
    if z==0:
        y = 0
        height = 0
    #cutsection = trialK(staff, staff_height, y, 20)
    cutsection = []; tempo = []
    for t in tqdm(range(1,50)):
        cut = trialK(staff, staff_height, y, t)
        if cut!=False:
            tempo.append(cut)
            tempo_idx = np.argmax(np.array([t[0] for t in tempo])) 
            cutsection.append(tempo[tempo_idx])
    heights = [c[3] for c in cutsection]
    mode_height = max(set(heights), key=heights.count)
    print(heights, "most height", mode_height)
    """
    max_prob_idx = np.argmax(np.array([q[0] for q in cutsection]))
    cutsection = cutsection[max_prob_idx]
    print(cutsection)
    """
    if len(heights)>0 and mode_height>0:
        height = mode_height
        y = y+height
        height_dataset.append(((z+1)*0.05, y))
        print("*******************")
        print("new height:", height,"at:", y)
        cv2.line(staff, (0,y), (staff.shape[1],y), (0,0,255), 2)
    else: pass
cv2.imwrite("cutsection.jpg", staff)
print(height_dataset)
x_set = np.array([h[1] for h in height_dataset])
y_set = np.array([h[0] for h in height_dataset])

slope, intercept, r, p, std_err = stats.linregress(x_set, y_set)

def fitLinear(x):
  return slope * x + intercept

mymodel = list(map(fitLinear, x_set))

plt.scatter(x_set, y_set)
plt.plot(x_set, mymodel)
plt.show()

print(staff.shape[0])
water_level = 1 - fitLinear(staff.shape[0])
print("water level", water_level)
"""
    ---- compare biggest contour with upper frame for check that contour is E ---- 
    if y-h<0:
        frame = cv2.threshold(cv2.resize(r3.copy()[0:y,x:x+w], (0,0), fx=2, fy=2), 128, 255, cv2.THRESH_OTSU)[1]
        num_rows_to_insert = r4.shape[0]-frame.shape[0]
        row_of_zeros = np.zeros((num_rows_to_insert, frame.shape[1]), dtype=frame.dtype)
        frame = np.insert(frame, 0, row_of_zeros, axis=0)
    else:
        frame = cv2.threshold(cv2.resize(r3.copy()[y-h:y,x:x+w], (0,0), fx=2, fy=2), 128, 255, cv2.THRESH_OTSU)[1]
    r5 = (np.array(r4, dtype="float32")+np.array(frame, dtype="float32"))/255
    white_sum = np.sum(r5)
    area = r5.shape[0]*r5.shape[1]
    print(white_sum/area)
"""
    
"""
    --- Wrap Perspective ---
    points = ct.reshape(4, 2)
    input_point = np.zeros((4, 2), dtype="float32")

    point_sum = points.sum(axis=1)
    input_point[0] = points[np.argmin(point_sum)]
    input_point[3] = points[np.argmax(point_sum)]

    point_diff = np.diff(points, axis=1)
    input_point[1] = points[np.argmin(point_diff)]
    input_point[2] = points[np.argmax(point_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_point
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2))
    top_width = np.sqrt(((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2))
    right_width = np.sqrt(((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2))
    left_width = np.sqrt(((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2))

    max_width = point[2] #max(int(bottom_width), int(top_width))
    max_height = point[3] #max(int(left_width), int(right_width))
    
    convert = np.float32([[0,0], [max_width, 0], [0, max_height], [max_width, max_height]]) 
    print(input_point, convert)  
    matrix = cv2.getPerspectiveTransform(input_point, convert)
    img3 = img2.copy()
    img_out = cv2.warpPerspective(img3, matrix, (max_width, max_height))    
"""




