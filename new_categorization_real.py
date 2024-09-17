import cv2 #type: ignore
import numpy as np
from picar import front_wheels, back_wheels #type: ignore
from picamera2 import Picamera2 #type: ignore
# from picamera2 import Preview
import picar #type: ignore
import time
from time import sleep
import new_screen_centering 

camera = Picamera2()
config = camera.create_still_configuration(main={"size": (1280, 960)}, lores={"size": (640, 480)}, encode="lores")
camera.configure(config)
camera.start()

camera.capture_file("test.png")
image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)#[820:2460, 616:1848]
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (7, 7), 0)

edges = cv2.Canny(blurred, 50, 150)

contours,_ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea)

def is_circle(contour):
    area = cv2.contourArea(contour)
    if area < 500:
        print("area:", area)
        return False
    perimiter = cv2.arcLength(contour, True)
    if perimiter == 0:
        return False
    circularity = 4 * np.pi * (area/(perimiter * perimiter))
    if circularity < 0.6:
        print("circularity:", circularity)
        return False
    (x, y), (width, height), angle = cv2.minAreaRect(contour)
    aspect_ratio = min(width, height) / max(width, height)
    if aspect_ratio < 0.85:
        print("ratio:", aspect_ratio)
        return False
    print("passed:", area, circularity, aspect_ratio)
    return True

stimuli = []
for i in contours:
    if is_circle(i):
        stimuli.append(i)
        cv2.drawContours(image, i, -1, (0, 255, 0), 2)
print(len(stimuli))

if len(stimuli) >= 2:
    x1, y1, w1, h1 = cv2.boundingRect(stimuli[0])
    x2, y2, w2, h2 = cv2.boundingRect(stimuli[1])
    mask1 = np.zeros_like(image, np.uint8)
    mask2 = np.zeros_like(image, np.uint8)

    # draw the contours on the masks
    cv2.drawContours(mask1, [stimuli[0]], -1, color=255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [stimuli[1]], -1, color=255, thickness=cv2.FILLED)

    image1 = cv2.bitwise_and(image, image, mask=mask1)
    image2 = cv2.bitwise_and(image, image, mask=mask2)
    value1 = np.sum(image1) / cv2.countNonZero(mask1)
    value2 = np.sum(image2) / cv2.countNonZero(mask2)
    if x1 < x2:
        print("test1")
        stimulus_1 = 1 - value1/255
        stimulus_2 = 1 - value2/255
    else:
        print("test2")
        stimulus_1 = 1 - value2/255
        stimulus_2 = 1 - value1/255
else: 
    stimulus_1 = 15 #can change these temporarily
    stimulus_2 = 25

print(stimulus_1, stimulus_2)

cv2.imwrite("result.png", image)

# Model Implementation

# Setting constants for the attentional model

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

import matplotlib.pyplot as plt #type: ignore
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter #type: ignore

def sigmoidFunction(x, a=1, b=1, c=0.5, m=5):
    """compute sigmoid values.
    
    Parameters:
    - x: The input value or array of values.
    - a: Minimum value of the sigmoid.
    - b: Range of the sigmoid.
    - m: The steepness of the sigmoid.
    - c: The midpoint of the sigmoid.
    
    Returns:
    - Sigmoid function values."""
    return a + (b*(x**m))/((x**m) + (c**m))

def sigmoid2(x, a=a, b=b, c=L50, m=k, sOut=0, sIn=0):
    """compute sigmoid values.
    
    Parameters:
    - x: The input value or array of values.
    - a: Minimum value of the sigmoid.
    - b: Range of the sigmoid.
    - m: The steepness of the sigmoid.
    - c: The midpoint of the sigmoid.
    - sOut: The output suppression factor of the sigmoid.
    - sIn: The input suppression factor of the sigmoid.
    
    Returns:
    - Sigmoid function values."""
    return (1/(sOut+1))*(a/(sIn+1) + b*(x**m/(x**m + c**m + sIn**m)))

# Defining stimuli
stim1 = []
stim2 = []

def lateral_inhibition(stim1, stim2 = np.linspace(0, 1, 101,), dIn = 0.5, dOut = 0.5):
    # Compute the activity of the inhibitory neurons
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2

    # Compute the activity of the excitatory neurons
    exc1 = sigmoid2(stim1, sIn = sIn2, sOut = sOut2)
    exc2 = sigmoid2(stim2, sIn = sIn1, sOut = sOut1)

    return exc1, exc2, inh1, inh2

def reciprocal_inhibition(stim1, stim2 = np.linspace(0, 1, 101), dIn = 0.5, dOut = 0.5, rIn = 0.5, rOut = 0.5):
    # Compute the activity of inhibitory neurons
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2

    # Initial inhibition factor at T = 1s
    inh1_t_1 = sigmoid2(0)
    inh2_t_1 = sigmoid2(0)

    # Max amount of difference allowed between original inh vs. opposing inh
    threshold = 10**-10
    timesteps = 0
    diff1 = np.array([1])
    diff2 = np.array([1])

    # Loop until threshold is met
    while np.any(diff1 > threshold) or np.any(diff2 > threshold):
        # Calculate new inhibition factors 
        iIn_1 = rIn * inh2_t_1
        iOut_1 = rOut* inh2_t_1
        iIn_2 = rIn * inh1_t_1
        iOut_2 = rOut * inh1_t_1

        # Reassign the original inh value with the newly calculated factors 
        inh1 = sigmoid2(stim1, sIn = iIn_1, sOut = iOut_1)
        inh2 = sigmoid2(stim2, sIn = iIn_2, sOut = iOut_2)

        # Calculate the difference between the old inh and the new inh factors 
        diff1 = np.abs(inh1_t_1 - inh1)
        diff2 = np.abs(inh2_t_1 - inh2)

        # Reassign the old inh factors with the newly calculated factors 
        inh1_t_1 = inh1
        inh2_t_1 = inh2
    
    # Compute the suppression factors
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2

    # Compute the activity of the excitatory neurons 
    exc1 = sigmoid2(stim1, sIn = sIn2, sOut = sOut2)
    exc2 = sigmoid2(stim2, sIn = sIn1, sOut = sOut2)

    return exc1, exc2, inh1, inh2

def self_inhibition(stim1, stim2 = np.linspace(0, 1, 101), dIn = 0.5, dOut = 0.5, rIn = 0.5, rOut = 0.5):
    # Compute the activity of inhibitory neurons
    inh1 = sigmoid2(stim1)
    inh2 = sigmoid2(stim2)

    # Compute suppression factors 
    sIn1 = dIn * inh1; sOut1 = dOut * inh1
    sIn2 = dIn * inh2; sOut2 = dOut * inh2

    # Initial inhibition factor at T = 1 second
    inh1_t_1 = sigmoid2(0)
    inh2_t_1 = sigmoid2(0)

    # Max amount of difference allowed between the original inh vs. the opposing inh
    threshold = 10**-10
    timesteps = 0
    diff1 = np.array([1])
    diff2 = np.array([1])

    # Loop until threshold is met
    while np.any(diff1 > threshold) or np.any(diff2 > threshold):
        # Calculate new inhibition factors 
        iIn_1 = rIn * inh2_t_1
        iOut_1 = rOut* inh2_t_1
        iIn_2 = rIn * inh1_t_1
        iOut_2 = rOut * inh1_t_1

        # Reassign the original inh value with the newly calculated factors 
        inh1 = sigmoid2(stim1, sIn = iIn_1, sOut = iOut_1)
        inh2 = sigmoid2(stim2, sIn = iIn_2, sOut = iOut_2)

        # Calculate the difference between the old inh and the new inh factors 
        diff1 = np.abs(inh1_t_1 - inh1)
        diff2 = np.abs(inh2_t_1 - inh2)

        # Reassign the old inh factors with the newly calculated factors 
        inh1_t_1 = inh1
        inh2_t_1 = inh2

    # Compute the suppression factors
    sIn1 = dIn * (inh1+inh2); sOut1 = dOut * (inh1+inh2)
    sIn2 = dIn * (inh2+inh1); sOut2 = dOut * (inh2+inh1)

    # Compute the activity of the excitatory neurons 
    exc1 = sigmoid2(stim1, sIn = sIn2, sOut = sOut2)
    exc2 = sigmoid2(stim2, sIn = sIn1, sOut = sOut2)

    return exc1, exc2, inh1, inh2

def addNoise(input, fanoFactor = 0):
    output = input + np.sqrt(input * fanoFactor) * np.random.randn(*input.shape)
    return output

def noiseFitFunction(exc1, exc2, ff, iters):
    target1 = np.zeros((iters, exc1.shape[0]))
    for i in range(iters):
        exc1_noisy = addNoise(exc1, ff)
        exc2_noisy = addNoise(exc2, ff)
        target1[i] = (exc1_noisy > exc2_noisy).astype(int)
    score1 = np.mean(target1, axis = 0)
    p0 = [min(score1), max(score1)-min(score1), np.mean(stim2), 1]
    np.random.seed(0)
    p_fit,_ = curve_fit(sigmoidFunction, stim2, score1, p0, method='dogbox')
    a1, b1, c1, m1 = p_fit
    score1_fit = sigmoidFunction(stim2, a1, b1, c1, m1)

    # Transition range
    tr_idx = np.zeros((2,), dtype=int) # start and end point indices of the transition range
    tr_sep = np.zeros((2,)) # start and end points of the transition range

    tr_idx[0] = np.argmax(score1_fit < 0.9)
    tr_idx[1] = np.argmax(score1_fit < 0.1)

    tr_sep[0] = stim2[tr_idx[0]]
    tr_sep[1] = stim2[tr_idx[1]]

    tr = tr_sep[1] - tr_sep[0]

    return score1, score1_fit, tr_sep, tr

# stimulus strengths
# Processing the strength of stimuli based on the detected objects 
stimulus_strength = []
print('stimulus strength 1: {0}'.format(stimulus_1))
print('stimulus strength 2: {0}'.format(stimulus_2))

fano_factor = 0.0
exc1, exc2,_,_ = reciprocal_inhibition(np.array(stimulus_1), np.array(stimulus_2), dIn = d_in, dOut = d_out, rIn = r_in, rOut = r_out)

OT_t1_1 = addNoise(exc1, fanoFactor = fano_factor)
OT_t1_2 = addNoise(exc2, fanoFactor = fano_factor)

print('final excitatory unit 1 activity: {0}'.format(OT_t1_1))
print('final excitatory unit 2 activity: {0}'.format(OT_t1_2))

# Setting up the Picar for movement
# Moving the robot towards the winning stimulus
picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
bw.speed = 0

# Determining the direction of movement based on the dominant stimulus 
# Moving the car towards the stronger stimulus 
if OT_t1_1 > OT_t1_2: # Stimulus on the left is bigger
    print("left stimulus is bigger")
    fw.turn(90-60)
    time.sleep(1)
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
    print("right stimulus is bigger")
    fw.turn(90+60)
    time.sleep(1)
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