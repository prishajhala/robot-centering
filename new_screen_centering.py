import cv2 #type: ignore
import numpy as np
from picar import front_wheels, back_wheels # type: ignore
from picamera2 import Picamera2, Preview # type: ignore
import picar # type: ignore
import time

# Set up the picar and set its speed to 0

picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
bw.speed = 0

# Method that, once the green dot is found, either adjusts to the left or right direction

def adjustPosition():
	# Speed is increased to 30
	bw.speed = 30
	# The x and y position of the green dot relative to the center of the robot's field of view are found
	cx, cy = find_green_dot()   
	print(cx, cy)
	# If the green dot is less than 50 pixels away from the center in the x direction, it has adjusted and just returns true
	if abs(cx) < 50:
		bw.speed = 0
		return True
    # If the opposite is true, the robot will move in the direction of the green dot
	# The movement consists of turning in that direction, moving for half a second, turning back forwards, and moving back that same amount of time
	if cx > 0:
		print("turning right")
		fw.turn(90+30)
		t_end = time.time() + 0.5
		while time.time() < t_end:
			bw.backward()
		t_end = time.time() + 0.5
		fw.turn(90)
		while time.time() < t_end:
			bw.forward()
		bw.speed = 0
	else:
		print("turning left")
		fw.turn(90-30)
		t_end = time.time() + 0.5
		while time.time() < t_end:
			bw.backward()
		t_end = time.time() + 0.5
		fw.turn(90)
		while time.time() < t_end:
			bw.forward()
		bw.speed = 0
	return False

# This is a method to find the green dot on the center of the laptop screen
def find_green_dot():
	# The image test.png is read by the program
	image = cv2.imread("test.png")
	
	# The image is converted to HSV form, where colors can be filtered
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	# Ranges for the shade of green are set, this may need to be adjusted depending on the lighting
	lower_green = np.array([30, 40, 40])
	upper_green = np.array([100, 255, 255])
	
	# A green mask is created
	green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
	
	# Anything present in the mask is contoured
	contours,_ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	cx = 0
	cy = 0
	
	# This loop goes through all the contours found
	for contour in contours:
		area = cv2.contourArea(contour)
		# If the area of the contour is within the desired range and the object is circular enough, a contour is drawn
		if 25 <= area <= 500:
			M = cv2.moments(contour)
			if M["m00"] != 0:
				#The x value and y value of the conour are taken
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
			
				cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
	
	# An image is written with the contours drawn on
	cv2.imwrite('output_with_contours.jpg', image)
	# The x and y values relative to the center are returned
	return cx-320, cy-240