import sys
import cv2 #type: ignore 
import numpy as np
import math 

"""
bw_circles = cv2.imread("/Users/prishajhala/Downloads/PiCar/Most recent code/lotsa circles.png")
gray = cv2.cvtColor(bw_circles, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# Param1/param2 use Canny edge detection methods to detect the edges of the circles 
# Param2 is the accumulator threshold for the circle centers - higher value means that fewer false, more strong detections 
# Changed param2 to < 100 to detect the smaller circle, changed the minDist to < 20 for smaller distances between circles 
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1, param1=45, param2=90, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0,255,0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0,0,255), 3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows
"""

def preprocessed_image(image_path):
    #image = cv2.imread("/Users/prishajhala/Downloads/PiCar/Most recent code/bw circles.png")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.medianBlur(gray, 5)
    return blurred_img

def detect_circles(preprocessed_image):
    circles = cv2.HoughCircles(preprocessed_image, cv2.HOUGH_GRADIENT, 1, 1, param1=45, param2=90, minRadius=0, maxRadius=0)
    return circles

def calculate_distance(circle1, circle2):
    x1, y1 = circle1[0], circle1[1]
    x2, y2 = circle2[0], circle2[1]
    center_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # print("distance between circle centers: ", center_dist)
    return center_dist

def edge_distance(circle1, circle2):
    r1, r2 = circle1[2], circle2[2]
    center_dist1 = calculate_distance(circle1, circle2)
    edge_dist = center_dist1 - (r1 + r2)
    # print("distance between circle edges: ", edge_dist)
    return edge_dist

preprocess_img = preprocessed_image("/Users/prishajhala/Downloads/PiCar/Most recent code/bw circles.png")
circles = detect_circles(preprocess_img)
if circles is not None:
    circles = np.uint16(np.around(circles))
    if len(circles[0]) >= 2:
        circle1 = circles[0][0]
        circle2 = circles[0][1]
        dist_center = calculate_distance(circle1, circle2)
        dist_edge = edge_distance(circle1, circle2)
        print("Distance between 2 circle edges: ", dist_edge)
        print("Distance between 2 circle centers:", dist_center)
        original_img = cv2.imread("/Users/prishajhala/Downloads/PiCar/Most recent code/bw circles.png")
        for i in circles[0, :]:
            cv2.circle(original_img, (i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(original_img, (i[0], i[1]), 2, (0,0,255), 3)
            
cv2.imshow('detected circles', original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def main(args):
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))