import cv2 #type: ignore
import numpy as np
from picar import front_wheels, back_wheels # type: ignore
from picamera2 import Picamera2, Preview # type: ignore
import picar # type: ignore
import time
from time import sleep
import new_screen_centering

# Createa picamera object
camera = Picamera2()
# Set the resolution and image size of the camera through its configurations
config = camera.create_still_configuration(main={"size": (1280,960)}, lores={"size": (640,480)}, encode="lores")
camera.configure(config)
# Start the camera
camera.start()

# Capture an image and save it in grayscale
camera.capture_file("test.png")
image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)#[820:2460, 616:1848]
# blur the captured image to remove noise
blurred = cv2.GaussianBlur(image, (7,7), 0)
# extract the edges from said image
edges = cv2.Canny(blurred, 50, 150)

# Using openCV's build in methods, find the most prominent contours in the image
contours,_ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Sort the contours by area
contours = sorted(contours, key=cv2.contourArea)

# A method to determine whether a contour is a circle and matches the criteria for it to be a contour of a stimulus
def is_circle(contour):
	# Get the area of the given contour
	area = cv2.contourArea(contour)
	# If the area of the contour is below a certain threshold, it's determined not to be correct
	if area < 500:
		print("area was too small:", area)
		return False
	# Get the perimiter of the given contour
	perimiter = cv2.arcLength(contour, True)
	# If the shape isn't continuous with a nonzero perimiter, it isn't the correct contour
	if perimiter == 0:
		return False
	# Circularity is a value that determines how circular an object is by determining the ratio of its area to the perimiter squared
	# Circularity in a perfect circle is 1, so the closer it is to 1 the more circular the contour is
	circularity = 4 * np.pi * (area / (perimiter * perimiter))
	# If the shape doesn't hit a circularity threshold, it's not the correct contour
	if circularity < 0.6:
		print("circularity was too low:", circularity)
		return False
	# The smallest rectangle that encompasses the contour should logically have a height to width ratio close to 1 if it's a circle
	(x,y), (width, height), angle = cv2.minAreaRect(contour)
	# This determines the aspect ratio, which divides the smaller of the width and heights by the larger of the width and heights
	aspect_ratio = min(width, height) / max(width, height)
	# If the rectangle has an aspect ratio below a certain threshold it's determined not to be a circle
	if aspect_ratio < 0.85:
		print("ratio was to small:", aspect_ratio)
		return False
	# If all these chekcks are passed, the shape is then determined to be a circle and the method returns true
	print("passed:",area,circularity,aspect_ratio)
	return True

# This is a list of all contours that pass the test
stimuli = []
for i in contours:
	if is_circle(i):
		# The shape is added to the list when it passes the test
		stimuli.append(i)
		# The passing contour is drawn onto the image
		cv2.drawContours(image, i, -1, (0, 255, 0), 2)

# If there are enough contours (just has to be greater than 2, since even if other objects pass the test the two largest contours are almost always the two intended circles)
if len(stimuli) >= 2:
	# The bounding rectangles of the two objects are determined
	x1,y1,w1,h1 = cv2.boundingRect(stimuli[0])
	x2,y2,w2,h2 = cv2.boundingRect(stimuli[1])

	# A mask the size of the original image is created
	mask1 = np.zeros_like(image, np.uint8)
	mask2 = np.zeros_like(image, np.uint8)

    # Draw the contours on the masks
	cv2.drawContours(mask1, [stimuli[0]], -1, color=255, thickness=cv2.FILLED)
	cv2.drawContours(mask2, [stimuli[1]], -1, color=255, thickness=cv2.FILLED)

    # Calculate the mean darkness of the pixels in each image by adding all their value and normalizing the values
	image1 = cv2.bitwise_and(image, image, mask=mask1)
	image2 = cv2.bitwise_and(image, image, mask=mask2)

	value1 = np.sum(image1) / cv2.countNonZero(mask1)
	value2 = np.sum(image2) / cv2.countNonZero(mask2)

	# These conditionals are meant so that the contour with the smallest x value, or the x value, is stimulus 1 for the sake of simplicity
	# The stimulus values are scaled on a scale of 0 to 1, with 1 being the darkest and 1 being the lightest
	if x1 < x2:
		stimulus_1 = 1 - value1/255
		stimulus_2 = 1 - value2/255
	else:
		stimulus_1 = 1 - value2/255
		stimulus_2 = 1 - value1/255
else:
	# Done so if no contours are recognized there's a value so that errors aren't given
	stimulus_1 = 0
	stimulus_2 = 0

cv2.imwrite("result.png", image)

# model implementation
# Setting constants for the attentional model.
# constants
r_out = 0.01
r_in = 0.8

m = 5
h = 30
s50 = 8
k = 7

d_out = 0.05
d_in = 1

a = 5.3
b = 22.2
L50 = 11.6

# determining baseline inhibitory activity (i.e. inhibitory unit activity at t0)
inhibition_tminus1 = m
predifference = 1

# Implementing a loop to simulate dynamic inhibitory activity over time.
while predifference > 0.05:
    # dependent on inhibition_tminus1_2
    i_in = r_in * inhibition_tminus1
    i_out = r_out * inhibition_tminus1
   
    # inhibitory unit activity at current time step
    inhibition_t0 = (1 / (i_out + 1)) * ((m / (i_in + 1)))
   
    # computing % change in inhibitory neuron activity from previous time step
    predifference = abs(1 - (inhibition_t0 / inhibition_tminus1))
   
    # updating inhibitory unit activity
    inhibition_tminus1 = inhibition_t0
   
# print('baseline inhibitory activity: {0}'.format(inhibition_t0))

inhibition_t0_1 = inhibition_t0
inhibition_t0_2 = inhibition_t0

# stimulus strengths
# Processing the strength of stimuli based on the detected objects.
stimulus_strength = []

print('stimulus strength 1: {0}'.format(stimulus_1))
print('stimulus strength 2: {0}'.format(stimulus_2))

fano_factor = 0.25

# Simulating the effect of each stimulus on the attentional model.
for k in range(101):
    # dependent on inhibition_t0_2
    i_in_1 = r_in * inhibition_t0_2
    i_out_1 = r_out * inhibition_t0_2
   
    # dependent on inhibition_t0_1
    i_in_2 = r_in * inhibition_t0_1
    i_out_2 = r_out * inhibition_t0_1
   
    # inhibitory unit activity at current time step
    inhibition_t1_1 = (1 / (i_out_1 + 1)) * ((m / (i_in_1 + 1)) + h * ((stimulus_1 ** k) / ((stimulus_1 ** k) + (s50 ** k) + (i_in_1 ** k))))
    inhibition_t1_2 = (1 / (i_out_2 + 1)) * ((m / (i_in_2 + 1)) + h * ((stimulus_2 ** k) / ((stimulus_2 ** k) + (s50 ** k) + (i_in_2 ** k))))
   
    # dependent on inhibition_t1_2
    s_in_1 = d_in * (inhibition_t1_2 + inhibition_t1_1)
    s_out_1 = d_out * (inhibition_t1_2 + inhibition_t1_1)
   
    # dependent on inhibition_t1_1
    s_in_2 = d_in * (inhibition_t1_1 + inhibition_t1_2)
    s_out_2 = d_out * (inhibition_t1_1 + inhibition_t1_2)
   
    # OT neuron activity at current time step
    OT_t1_1 = (1 / (s_out_1 + 1)) * ((a / (s_in_1 + 1)) + b * (np.square(stimulus_1) / (np.square(stimulus_1) + np.square(L50) + np.square(s_in_1))))
    OT_t1_2 = (1 / (s_out_2 + 1)) * ((a / (s_in_2 + 1)) + b * (np.square(stimulus_2) / (np.square(stimulus_2) + np.square(L50) + np.square(s_in_2))))
    
    # adding noise 
    OT_t1_1 = OT_t1_1 + (np.sqrt(OT_t1_1 * fano_factor) * np.random.normal())
    OT_t1_2 = OT_t1_2 + (np.sqrt(OT_t1_2 * fano_factor) * np.random.normal())
    
    # updating inhibitory unit activity
    inhibition_t0_1 = inhibition_t1_1
    inhibition_t0_2 = inhibition_t1_2
   
   
print('final excitatory unit 1 activity: {0}'.format(OT_t1_1))
print('final excitatory unit 2 activity: {0}'.format(OT_t1_2))

f = open("output.txt", "w")
f.write(OT_t1_1, OT_t1_2)
f.close() 

# Setting up the PiCar for movement.
# moving the robot towards the winning stimulus
picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
bw.speed = 0

# Determining the direction of movement based on the dominant stimulus.
# Moving the car towards the stronger stimulus.
if OT_t1_1 > OT_t1_2: # stimulus on the left is bigger
	print("left stimulus")
	print(OT_t1_1, OT_t1_2)
	fw.turn(90-30)
	bw.speed = 30
	t_end = time.time() + 1
	while time.time() < t_end:
		bw.backward()
	bw.speed = 0
	fw.turn(90)
	bw.speed = 30
	t_end = time.time() + 1
	while time.time() < t_end:
		bw.forward()
	bw.speed = 0
    
   
else: # stimulus on the right is bigger
	print("right stimulus")
	print(OT_t1_1, OT_t1_2)
	fw.turn(90+30)
	bw.speed = 30
	t_end = time.time() + 1
	while time.time() < t_end:
		bw.backward()
	bw.speed = 0
	fw.turn(90)
	bw.speed = 30
	t_end = time.time() + 1
	while time.time() < t_end:
		bw.forward()
	bw.speed = 0

sleep(1)

while new_screen_centering.adjustPosition() != True:
	camera.capture_file("test.png")