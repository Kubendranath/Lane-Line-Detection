import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

path = "test2.mp4"

cap = cv2.VideoCapture(path)
out = cv2.VideoWriter()


def make_coordinates(image , line_parameters):
  slope,intercept = line_parameters
  y1 = image.shape[0]
  y2 = int(y1*(3/5))
  x1 = int((y1-intercept)/slope)
  x2 = int((y2-intercept)/slope)
  return np.array([x1,y1,x2,y2])
def average_slope_image(image , lines):
  left_average = []
  right_average = []

  for line in lines:
    x1,y1,x2,y2 = line.reshape(4)
    parameters = np.polyfit((x1,x2) , (y1,y2) ,  1 )
    slope = parameters[0] 
    intercept = parameters[1]
    if slope<0:
      left_average.append((slope,intercept))
    else:
      right_average.append((slope,intercept))

  left_side_average = np.average(left_average , axis =0)
  right_side_average = np.average(right_average , axis =0)  
  left_line = make_coordinates(image , left_side_average)   
  right_line = make_coordinates(image , right_side_average)
  return np.array([left_line ,right_line])
def display_lines(image , lines):
  line_image = np.zeros_like(image)

  if lines is not None :
    for line in lines:
      x1,y1,x2,y2 = line.reshape(4)
      cv2.line(line_image,(x1,y1), (x2,y2) , (255,0,0),10)

    return line_image

def canny(image):
  gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray , (5,5) , 0)
  canny = cv2.Canny(blur,50,150)
  return canny


def region_of_interest(image):
  height = image.shape[0]
  width = image.shape[1]

  polygons = np.array([
      [(200,height),(1100,height),(550,250)]
      ])

  mask = np.zeros_like(image)
  cv2.fillPoly(mask , polygons,255)
  masked_image = cv2.bitwise_and(image,mask)
  return masked_image
while(cap.isOpened()):
  ret,frame = cap.read()
  
  if ret == True:
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)  
    hough_image = cv2.HoughLinesP(cropped_image , 2,np.pi/180 , 5 , np.array([]),minLineLength = 40, maxLineGap = 5)
    averaged_lines = average_slope_image(frame , hough_image)
    lines = display_lines(frame,averaged_lines)
    result_image = cv2.addWeighted(frame , 0.8 , lines , 1 , 1)
    cv2.imshow("result",result_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
     break
  else:
    break

cap.release()
cv2.destroyAllWindows()

