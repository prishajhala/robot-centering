import cv2 #type: ignore
import numpy as np

planets = cv2.imread("/Users/prishajhala/Downloads/PiCar/Most recent code/planets.png")
gray = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
rows = img.shape[0]
# Param1/param2 use canny edge detection to find the edges of the circles - 
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=1, minRadius=1, maxRadius=30)
"""
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        cv2.circle(planets, center, 1, (0,100,100), 3)
        radius = i[2]
        cv2.circle(planets, center, radius, (255,00,255), 3)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=0, maxRadius=0)
cv2.circle(planets, (120,50), 30, (0,0,255), 2)

if circles is not None:
    circles = np.uint8(np.around(circles))
    for i in circles[0 :]:
        cv2.circle(planets, (i[0], i[1]), 2, (0,0,255), 2)
        cv2.circle(planets, (i[0], i[1]), 2, (0,0,255), 3)

circles = np.uint8(np.around(circles))
for i in circles[0, :]:
    cv2.circle(planets, i[0], i[1], i[2], (0,0,255), 2)
    cv2.circle(planets, i[0], i[1], 2, (0,255,0), 3)
"""

cv2.imshow("circle detection", planets)
cv2.waitKey(0)
cv2.destroyAllWindows()

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))