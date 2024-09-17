import sys
import math 
import time
from time import sleep
from picamera2 import Picamera2 #type: ignore
import scipy.ndimage as scimg
import numpy as np
import matplotlib.pyplot as plt #type: ignore
from matplotlib.collections import PatchCollection #type: ignore
from matplotlib.patches import Rectangle #type: ignore
from matplotlib.ticker import FormatStrFormatter #type: ignore
from sklearn.cluster import DBSCAN #type: ignore
import picar #type: ignore
from picar import front_wheels, back_wheels #type: ignore
import cv2 #type: ignore
# import new_screen_centering
from scipy.optimize import curve_fit 

# Picamera setup
h = 640
cam_res = (int(h), int(0.75 * h))
cam_res = (int(32 * np.floor(cam_res[0]/32)), int(16 * np.floor(cam_res[1]/16)))

"""
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": cam_res}))
picam2.start()
"""
camera = Picamera2()
config = camera.create_still_configuration(main={"size":(1280, 960)}, lores={"size": (640, 480)}, encode="lores")
camera.configure(config)
camera.start()

# Preallocating image variables
data = np.empty((cam_res[1], cam_res[0], 3), dtype=np.uint8)

# Different edge detection methods
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
t1 = time.time()
data = camera.capture_array("main")

# TODO: Crop data appropriately 

data = data[100:300, 100:500, 0:3]

fig2, ax2 = plt.subplots(1, 1, figsize=(12,8))
ax2.imshow(data)

scale_val = 0.25
min_samps = 20
leaf_sz = 15
max_dxdy = 35
gaus = scimg.fourier_gaussian(scimg.zoom(np.mean(data, 2), scale_val), sigma = 0.01)
x, y = np.meshgrid(np.arange(0, np.shape(data)[1], 1/scale_val), np.arange(0, np.shape(data)[0], 1/scale_val))

# Canny method without angle
can_x = scimg.prewitt(gaus, axis = 0)
can_y = scimg.prewitt(gaus, axis = 1)
can = np.hypot(can_x, can_y)
ax[0].pcolormesh(x, y, gaus)

# Pulling out object edges
bin_size = 100
percent_cutoff = 0.018
hist_vec = np.histogram(can.ravel(), bins = bin_size)
hist_x, hist_y = hist_vec[0], hist_vec[1]
for ii in range(np.argmax(hist_x), bin_size):
    hist_max = hist_y[ii]
    if hist_x[ii] < percent_cutoff * np.max(hist_x):
        break

# sklearn section for clustering
x_cluster = x[can > hist_max]
y_cluster = y[can > hist_max]
x_scaled = np.where(can > hist_max, x, 0)
y_scaled = np.where(can > hist_max, y, 0)
scat_pts = []
for ii, jj in zip(x_cluster, y_cluster):
    scat_pts.append((ii, jj))

# Clustering analysis for object detecton
clustering = DBSCAN(eps=max_dxdy, min_samples = min_samps, algorithm='ball_tree', leaf_size=leaf_sz).fit(scat_pts)
nn_time = time.time() - t1
stimulus_strength_dict = {}

# Looping through each individual object
for ii in np.unique(clustering.labels_):
    if ii == -1:
        continue
    clus_dat = np.where(clustering.labels_==ii)

    x_pts = x_cluster[clus_dat]
    y_pts = y_cluster[clus_dat]
    cent_mass = (np.mean(x_pts), np.mean(y_pts))
    if cent_mass[0] < np.min(x) + 10 or cent_mass[0] > np.max(x) - 10 or cent_mass[1] < np.min(y) + 10 or cent_mass[1] > np.max(y) - 10:
        continue
    ax[1].plot(x_pts, y_pts, marker='.', linestyle='', label='Unrotated Scatter')

    # Rotation algorithm
    evals, evecs = np.linalg.eigh(np.cov(x_pts, y_pts))
    angle = np.arctan(evecs[0][1]/evecs[0][0])
    rot_vec = np.matmul(evecs.T, [x_pts, y_pts])

    # Rectangle algorithm 
    if angle < 0:
        rect_origin = (np.matmul(evecs, [np.min(rot_vec[0]), np.max(rot_vec[1])]))
    else:
        rect_origin = (np.matmul(evecs, [np.max(rot_vec[0]), np.min(rot_vec[1])]))
    
    rect_width = np.max(rot_vec[0]) - np.min(rot_vec[0])
    rect_height = np.max(rot_vec[1]) - np.min(rot_vec[1])
    obj_rect = Rectangle(rect_origin, rect_width, rect_height, angle = (angle/np.pi) * 180)
    pc = PatchCollection([obj_rect], facecolor="None", edgecolor='r', linewidth=2)
    ax2.add_collections(pc)
    ax2.annotate('{:.2f}$^\circ$ Rotation'.format((angle/np.pi) * 180), xy=(rect_origin), xytext=(0, 0), textcoords='offset points', bbox=dict(fc='white'))

    radius = (rect_width + rect_height)/4
    area = np.pi * np.square(radius)
    stimulus_strength_dict[int(rect_origin[0])] = area

ax[1].set_xlim(np.min(x), np.max(x))
ax[1].set_ylim(np.min(y), np.max(y))

fig2.savefig('rectangles_over_real_image.png', dpi = 200, facecolor=[252/255, 252/255, 252/255])

stimulus_strength = []
if stimulus_strength_dict:
    temp = min(stimulus_strength_dict)
    stimulus_strength.append(stimulus_strength_dict[temp])
    temp = max(stimulus_strength_dict)
    stimulus_strength.append(stimulus_strength_dict[temp])
    stimulus_strength[1] = stimulus_strength[1]/1200
    stimulus_strength[0] = stimulus_strength[0]/1200
    stimulus_1 = stimulus_strength[0]
    stimulus_2 = stimulus_strength[1]
else: # Change
    stimulus_1 = 0
    stimulus_2 = 0

stimulus_1_rounded = round(stimulus_1)
switch_values = [6.1, 6.2, 6.3, 6.4, 6.5, 6.5, 6.6, 6.6, 6.6, 6.7, 6.7, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 7, 7, 7, 7, 7.1, 7.1, 7.1, 7.1, 7.1, 7.2, 7.2, 7.2, 7.2]

picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
bw.speed = 0

# new_categorization_real.py implemented here 

camera.capture_file("test.png")
image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (7,7),0)

edges = cv2.Canny(blurred, 50, 150)

contours,_ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea)

def is_circle(contour):
    area = cv2.contourArea(contour)
    if area < 500:
        return False
    perimiter = cv2.arcLength(contour, True)
    if perimiter == 0:
        return False
    circularity = 4*np.pi*(area/(perimiter * perimiter))
    if circularity < 0.6:
        return False
    (x, y), (width, height), angle = cv2.minAreaRect(contour)
    aspect_ratio = min(width, height) / max(width, height)
    if aspect_ratio < 0.85:
        return False
    return True 

image = cv2.imread("test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

stimuli = []
for i in contours:
    if is_circle(i):
        stimuli.append(i)
        cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
    
print("length of stimuli:", len(stimuli))

if len(stimuli) >= 2:
    x1, y1, w1, h1 = cv2.boundingRect(stimuli[0])
    x2, y2, w2, h2 = cv2.boundingRect(stimuli[1])
    mask1 = np.zeros_like(image, np.uint8)
    mask2 = np.zeros_like(image, np.uint8)

    # Draw the contours on the masks
    cv2.drawContours(mask1, [stimuli[0]], -1, color=255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [stimuli[1]], -1, color=255, thickness=cv2.FILLED)

    image1 = cv2.bitwise_and(image, image, mask=mask1)
    image2 = cv2.bitwise_and(image, image, mask=mask2)
    value1 = np.sum(image1)/cv2.countNonZero(mask1)
    value2 = np.sum(image2)/cv2.countNonZero(mask2)

    if x1 < x2:
        print("test 1")
        stimulus_1 = 1 - value1/255
        stimulus_2 = 1 - value2/255
    else:
        print("test 2")
        stimulus_1 = 1 - value2/255
        stimulus_2 = 1 - value1/255
else: # Change these values for testing
    stimulus_1 = 15
    stimulus_2 = 25

print(stimulus_1, stimulus_2)

cv2.imwrite("result.png", image)

# Implementing the attentional model

# Setting the constants for the model
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

# Computing sigmoid values
def sigmoidFunction(x, a=1, b=1, c=0.5, m=5):
    """
    Parameters:
    - x: The input value or array of values
    - a: Minimum value of the sigmoid
    - b: Range of the sigmoid
    - c: The midpoint of the sigmoid
    - m: The steepness of the sigmoid 

    Returns:
    - Sigmoid function value(s)
    """

    # Debugging
    print(a + (b*(x**m))/((x**m) + (c**m)))

    return a + (b*(x**m))/((x**m) + (c**m))

def sigmoid2(x, a=a, b=b, c=L50, m=k, sOut=0, sIn=0):
    """
    Compute sigmoid values with input and output suppression factors

    Parameters:
    - x: The input value or array of values
    - a: The minimum value of the sigmoid
    - b: Range of the sigmoid
    - c: The midpoint of the sigmoid
    - m: The steepness of the sigmoid
    - sIn: The input suppression factor of the sigmoid
    - sOut: The output suppression factor of the sigmoid

    Returns:
    - Sigmoid function value(s)
    """

    # Debugging
    print((1/(sOut+1))*(a/(sIn+1)+b*(x**m/(x**m + c**m + sIn**m))))

    return (1/(sOut+1))*(a/(sIn+1)+b*(x**m/(x**m + c**m + sIn**m)))

# Defining stimuli
stim1 = []
stim2 = []

def lateral_inhibition(stim1, stim2 = np.linspace(0, 1, 101), dIn=0.5, dOut=0.5):

    # Compute the activity of the inhibitory neurons
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2 

    # Compute the activity of the excitatory neurons 
    exc1 = sigmoid2(stim1, sIn=sIn2, sOut=sOut2)
    exc2 = sigmoid2(stim2, sIn=sIn1, sOut=sOut1)

    # Debugging
    print(exc1, exc2, inh1, inh2)

    return exc1, exc2, inh1, inh2

def reciprocal_inhibition(stim1, stim2=np.linspace(0, 1, 101), dIn=0.5, dOut=0.5, rIn=0.5, rOut=0.5):

    # Compute the activity of the inhibitory neurons 
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2 

    # Initial inhibition factor at time = 1 sec 
    inh1_t_1 = sigmoid2(0)
    inh2_t_1 = sigmoid2(0)

    # Maximum amount of difference allowed between the original inh and the opposing inh 
    threshold = 10**-10
    timesteps = 0
    diff1 = np.array([1])
    diff2 = np.array([1])

    # Loop until threshold is met 
    while np.any(diff1 > threshold) or np.any(diff2 > threshold):
        # Calculate new inhibition factors
        iIn_1 = rIn * inh2_t_1
        iOut_1 = rOut * inh2_t_1
        iIn_2 = rIn * inh1_t_1
        iOut_2 = rOut * inh1_t_1

        # Reassign the original inh values with the newly calculated factors 
        inh1 = sigmoid2(stim1, sIn=iIn_1, sOut=iOut_1)
        inh2 = sigmoid2(stim2, sIn=iIn_2, sOut=iOut_2)

        # Calculate the difference between the old inh and new inh factors 
        diff1 = np.abs(inh1_t_1 - inh1)
        diff2 = np.abs(inh2_t_1 - inh2) 
        
        # Reassign the old inh factors with the newly calculated factors 
        inh1_t_1 = inh1
        inh2_t_1 = inh2

        # Compute the suppression factors 
        sIn1 = dIn * inh1; sOut1 = dOut * inh1
        sIn2 = dIn * inh2; sOut2 = dOut * inh2 

        # Compute the activity of the excitatory neurons 
        exc1 = sigmoid2(stim1, sIn=sIn2, sOut=sOut2)
        exc2 = sigmoid2(stim2, sIn=sIn1, sOut=sOut1)
    
    # Debugging
    print(exc1, exc2, inh1, inh2)

    return exc1, exc2, inh1, inh2

def self_inhibition(stim1, stim2=np.linspace(0, 1, 101), dIn=0.5, dOut=0.5, rIn=0.5, rOut=0.5):

    # Compute the activity of inhibitory neurons
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2

    # Initial inhibition factor at time = 1 sec
    inh1_t_1 = sigmoid2(0)
    inh2_t_1 = sigmoid2(0)

    # Maximum amount of difference allowed between the original inh  and opposing inh values
    threshold = 10**-10
    timesteps = 0
    diff1 = np.array([1])
    diff2 = np.array([1])

    # Loop until threshold is met
    while np.any(diff1 > threshold) or np.any(diff2 > threshold):
        # Calculate new inhibition factors
        iIn_1 = rIn * inh2_t_1
        iOut_1 = rOut * inh2_t_1
        iIn_2 = rIn * inh1_t_1
        iOut_2 = rOut * inh2_t_1

        # Reassign the original inh value with the newly calculated factors 
        inh1 = sigmoid2(stim1, sIn=iIn_1, sOut=iOut_1)
        inh2 = sigmoid2(stim2, sIn=iIn_2, sOut=iOut_2)

        # Calculate the difference between old inh and new inh factors
        diff1 = np.abs(inh1_t_1 - inh1)
        diff2 = np.abs(inh2_t_1 - inh2)

        # Reassign the old inh factors with the newly calculated factors 
        inh1_t_1 = inh1
        inh2_t_1 = inh2

        # Compute the suppression factors 
        sIn1 = dIn * (inh1 + inh2); sOut1 = dOut * (inh1 + inh2)
        sIn2 = dIn * (inh2 + inh1); sOut2 = dOut * (inh2 + inh1)

        # Compute the activity of the excitatory neurons 
        exc1 = sigmoid2(stim1, sIn=sIn2, sOut=sOut2)
        exc2 = sigmoid2(stim2, sIn=sIn1, sOut=sOut1)

    # Debugging
    print(exc1, exc2, inh1, inh2)

    return exc1, exc2, inh1, inh2

def addNoise(input, fanoFactor=0): # fanoFactor measures Fano noise (fluctuation of an electric charge) in ion detectors 

    output = input + np.sqrt(input * fanoFactor) * np.random.randn(*input.shape)

    # Debugging
    print(output)

    return output

def noiseFitFunction(exc1, exc2, ff, iters):

    target1 = np.zeros((iters, exc1.shape[0]))
    for i in range(iters):
        exc1_noisy = addNoise(exc1, ff)
        exc2_noisy = addNoise(exc2, ff)
        target1[i] = (exc1_noisy > exc2_noisy).astype(int)
        score1 = np.mean(target1, axis = 0)
        p0 = [min(score1), max(score1) - min(score1), np.mean(stim2), 1]
        np.random.seed(0)
        p_fit,_ = curve_fit(sigmoidFunction, stim2, score1, p0, method='dogbox')
        a1, b1, c1, m1 = p_fit
        score1_fit = sigmoidFunction(stim2, a1, b1, c1, m1)

        # Transition range
        tr_idx = np.zeros((2,), dtype=int) # Start and end point indices of the transition range
        tr_sep = np.zeros((2,)) # Start and end points of the transition range 

        tr_idx[0] = np.argmax(score1_fit < 0.9)
        tr_idx[1] = np.argmax(score1_fit < 0.1)

        tr_sep[0] = stim2[tr_idx[0]]
        tr_sep[1] = stim2[tr_idx[0]]

        tr = tr_sep[1] - tr_sep[0]

        # Debugging 
        print('tr_idx', str(tr_idx))
        print('tr_sep', str(tr_sep))
        print('tr', str(tr))
        print(tr_idx[0])
        print(stim2[21])

    return score1, score1_fit, tr_sep, tr

# Stimulus strengths - Processing the strength of stimuli based on the detected objects
stimulus_strength = []
print('stimulus strength 1: {0}'.format(stimulus_1))
print('stimulus strength 2: {0}'.format(stimulus_2))

fano_factor = 0.0 # Change for testing 

exc1, exc2,_,_ = reciprocal_inhibition(np.array(stimulus_1), np.array(stimulus_2), dIn=d_in, dOut=d_out, rIn=r_in, rOut=r_out)
OT_t1_1 = addNoise(exc1, fanoFactor = fano_factor)
print('final excitatory unit 1 activity:{0}'.format(OT_t1_1))
OT_t1_2 = addNoise(exc2, fanoFactor = fano_factor)
print('final excitatory unit 2 activity:{0}'.format(OT_t1_2))

# Setting up the PiCar for movement - Moving the robot towards the dominant stimulus
picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
bw.speed = 0

# Determining the direction of movement based on the dominant stimulus 
if OT_t1_1 > OT_t1_2: # Stimulus on the left is bigger
    print("left stimulus is bigger - turning left")
    print("final excitatory units activity: ", OT_t1_1, OT_t1_2)
    fw.turn(90-60) # Change angle depending on how much we want the robot to turn (45 degrees??)
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.backward()
    bw.speed = 0
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.forward()
    bw.speed = 0
    fw.turn(90)
else: # Stimulus on the right is bigger
    print("right stimulus is bigger - turning right")
    print("final excitatory units activity: ", OT_t1_1, OT_t1_2)
    fw.turn(90+60) # Change angle depending on how much we want the robot to turn (45 degrees??)
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.backward()
    bw.speed = 0
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.forward()
    bw.speed = 0
    fw.turn(90)

# Setting up a Hough Circle Transform to move the robot back to its original position
# Get Hough Transform to use the final excitatory units to move towards and from the bigger circle
def preprocessed_image(image_path):
    #image = cv2.imread("/Users/prishajhala/Downloads/PiCar/Most recent code/bw circles.png")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.medianBlur(gray, 5)
    return blurred_img

def detect_circles(preprocessed_image):
    circles = cv2.HoughCircles(preprocessed_image, cv2.HOUGH_GRADIENT, 1, 1, param1=45, param2=90, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.round(circles[0, :]),np.astype("int")
        if len(circles) == 2: # Detecting 2 circles
            return [(x, y, r) for (x, y, r) in circles]
    return None

def midpoint(c1, c2):
    x1, y1, _ = c1
    x2, y2, _ = c2
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
    return midpoint 

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

# Moving the robot back to the center of the circles
if OT_t1_1 > OT_t1_2: # Stimulus on the left is bigger
    print("turning right")
    print("final excitatory units activity: ", OT_t1_1, OT_t1_2)
    fw.turn(90+60) # Change angle depending on how much we want the robot to turn (45 degrees??)
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.backward()
    bw.speed = 0
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.forward()
    bw.speed = 0
    fw.turn(90)
else: # Stimulus on the right is bigger
    print("turning left")
    print("final excitatory units activity: ", OT_t1_1, OT_t1_2)
    fw.turn(90+60) # Change angle depending on how much we want the robot to turn (45 degrees??)
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.backward()
    bw.speed = 0
    bw.speed = 30
    t_end = time.time() + 1
    while time.time() < t_end:
        bw.forward()
    bw.speed = 0
    fw.turn(90)

camera.stop()

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))