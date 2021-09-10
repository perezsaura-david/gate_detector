"""
Code in python for aruco detection and pose estimation
"""

import cv2, os
from tqdm import tqdm
from cv2 import aruco
import numpy as np

ARUCO_SIZE = 0.175 #meters
N_GATES = 4
GATE_SIZE = 1.5 #meters

# image size (1280x720)

cameraMatrix = np.array([[790.8989920514197, 0.0, 670.332791421756],[0.0, 789.6808338497912, 370.6481124492188], [0.0, 0.0, 1.0]])
distCoeffs   = np.array([-0.03448682771417174, -0.055932650937412745, 0.11969799783448262, -0.09163586323944228])

# Define the aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Define parameters for aruco detection
parameters = cv2.aruco.DetectorParameters_create()

def translate_to_center(tvec, rvec):
    
    # print('tvec',tvec)
    # print('rvec',rvec)
    rvec= rvec.reshape(3,1)
    tvec= tvec.reshape(3,1)
    
    r_out = np.zeros((3,1))
    t_out = np.zeros((3,1))
    t_gate = np.array([-GATE_SIZE/2,0,0])
    cv2.composeRT(r_out,t_gate,rvec,tvec,r_out,t_out)
    return t_out, r_out
    

# Load image
IMAGES_PATH = './Dataset/Data_Adam/gates/'

lines = []
for image_name in tqdm(os.listdir(IMAGES_PATH)):

    image = cv2.imread(IMAGES_PATH+image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if not ids:
        continue

    # Retrieve the 3d pose of each marker
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_SIZE, cameraMatrix, distCoeffs)

    # Draw the markers on the image
    image = cv2.aruco.drawDetectedMarkers(image, corners)
    # Draw the axes of each marker

    tvecs, rvecs = translate_to_center(tvecs, rvecs)                
    image = cv2.aruco.drawAxis(image, cameraMatrix, distCoeffs, rvecs, tvecs,0.1)
    
    line = f'{image_name} '
    for data in tvecs:
        line = line + f'{data[0]:.6f} '
    for data in rvecs:
        line = line + f'{data[0]:.6f} '
    line = line[:-1] + '\n'
    lines.append(line)
    
    
    # Display the image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
# cv2.destroyAllWindows()

with open('aruco_pose.txt', 'w') as f:
    f.writelines(lines)
