#!/usr/bin/etc python

import cv2
import numpy as np
from imutils import contours as contours1
import imutils
from  PIL import Image

Number='num.png'
img=cv2.imread(Number,cv2.IMREAD_COLOR)
copy_img=img.copy()
copy_img1 = img.copy()
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg',img2)
blur = cv2.GaussianBlur(img2,(3,3),0)
cv2.imwrite('blur.jpg',blur)
canny=cv2.Canny(img2,100,200)
cv2.imwrite('canny.jpg',canny)
contours,hierarchy  = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
box1=[]
f_count=0
select=0
plate_width=0

for i in range(len(contours)):
     cnt=contours[i]
     area = cv2.contourArea(cnt)
     x,y,w,h = cv2.boundingRect(cnt)
     rect_area=w*h  #area size
     aspect_ratio = float(w)/h # ratio = width/height
     #print("rect_area:",rect_area)
     #print("aspect_ratio :",aspect_ratio)
     #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
     #box1.append(cv2.boundingRect(cnt))
     if  (aspect_ratio>=0.1)and(aspect_ratio<=4.5)and(rect_area>=50)and(rect_area<=1000):
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
          box1.append(cv2.boundingRect(cnt))


for i in range(len(box1)):
     for j in range(len(box1)-(i+1)):
          if box1[j][0]>box1[j+1][0]:
               temp=box1[j]
               box1[j]=box1[j+1]
               box1[j+1]=temp


for m in range(len(box1)):
     count=0
     for n in range(m+1,(len(box1)-1)):
          delta_x=abs(box1[n+1][0]-box1[m][0])
          if delta_x > 150:
               break
          delta_y =abs(box1[n+1][1]-box1[m][1])
          if delta_x ==0:
               delta_x=1
          if delta_y ==0:
               delta_y=1
          gradient =float(delta_y) /float(delta_x)
          if gradient<0.25:
              count=count+1
     if count > f_count:
          select = m
          f_count = count
          plate_width=delta_x
cv2.imwrite('snake.jpg',img)


number_plate=copy_img[box1[select][1]-30:box1[select][3]+box1[select][1]+30,box1[select][0]-30:500+box1[select][0]]
resize_plate=cv2.resize(number_plate,None,fx=1.8,fy=1.8,interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
plate_gray=cv2.cvtColor(number_plate,cv2.COLOR_BGR2GRAY)
ret,th_plate = cv2.threshold(plate_gray,200,255,cv2.THRESH_BINARY)

cv2.imwrite('plate_th.jpg',th_plate)
kernel = np.ones((3,3),np.uint8)
er_plate = cv2.erode(th_plate,kernel,iterations=1)
er_invplate = er_plate
cv2.imwrite('er_plate.jpg',er_invplate)
#add
plat = er_invplate.copy()
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
threshed = cv2.morphologyEx(th_plate, cv2.MORPH_CLOSE, rect_kernel)
#cnts,hi_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
cnts = imutils.grab_contours(cnts)
for c in cnts:
     # compute the bounding box of the contour
     x1,y1,w1,h1 = cv2.boundingRect(c)
     #if w1>50 and h1>50:
      #    cv2.rectangle(threshed, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 5)
     #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

     # if the contour is sufficiently large, it must be a digit
     if w1 >= 10 and (h1 >= 30 and h1 <= 500):
          digitCnts.append(c)
print(len(digitCnts))
cv2.imwrite('result.jpg', threshed)
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9,
	(1, 1, 1, 0, 0, 1, 1) :0,
	(0, 1, 0, 0, 0, 1, 0) : 1,


}

digitCnts = contours1.sort_contours(digitCnts,method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = threshed[y:y + h, x:x + w]

	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)

	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)

	# loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)

		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1

	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(threshed, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(threshed, str(digit), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the digits
print(u"{}{}{}".format(*digits))
cv2.imwrite('result.jpg',threshed)
