import cv2
import numpy as np

def make_coordinates(img,line_param):
    slope,intercept=line_param
    y1=img.shape[0]
    y2=int(y1*3/5)
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(img,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        param=np.polyfit((x1, x2),(y1, y2),1)
        slope= param[0]
        intercept =param[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0)
    left_line=make_coordinates(img,left_fit_avg)
    right_line=make_coordinates(img,right_fit_avg)
    return np.array([left_line,right_line])



def canny(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def display_lines(img,lines):
    line_image=np.zeros_like(img)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def ROIF(img):
    height = img.shape[0]
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
        ])
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255)
    masked_img=cv2.bitwise_and(img,mask)
    return masked_img



img = cv2.imread('road.jpg')
laneimg= np.copy(img)
canny_img=canny(laneimg)
cropimg= ROIF(canny_img)
lines=cv2.HoughLinesP(cropimg,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
avgline=average_slope_intercept(laneimg,lines)
lineimg=display_lines(laneimg,avgline)
blendimg=cv2.addWeighted(laneimg,0.8,lineimg,1,1)
cv2.imshow("result", blendimg)
#plt.imshow(blendimg)
cv2.waitKey(0)
#plt.show() 


    
    
