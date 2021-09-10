"""
Code in python for aruco detection and pose estimation
"""

import cv2, os
from tqdm import tqdm
from cv2 import aruco
import numpy as np

# image size (1280x720)

cameraMatrix = np.array([[790.8989920514197, 0.0, 670.332791421756],[0.0, 789.6808338497912, 370.6481124492188], [0.0, 0.0, 1.0]])
distCoeffs   = np.array([-0.03448682771417174, -0.055932650937412745, 0.11969799783448262, -0.09163586323944228])


# Load image
IMAGES_PATH = './Dataset/Data_Adam/gates/'
ARUCO_TXT_FILE = './aruco_pose.txt'
PNP_TXT_FILE = './estimation_net.txt'

SHOW_AXIS = False
# SHOW_AXIS = True

aruco_lines = []
pnp_lines = []

with open(ARUCO_TXT_FILE) as f:
    aruco_lines = f.readlines()
with open(PNP_TXT_FILE) as f:
    pnp_lines = f.readlines()
    

def extract_common_images():
    aruco_images = []
    pnp_images = []
    for line in aruco_lines:
        aruco_images.append(line.split(' ')[0].strip())
    for line in pnp_lines:
        pnp_images.append(line.split(' ')[0].strip())
    common_images = list(set(aruco_images).intersection(pnp_images))
    print('Common images =',len(common_images))
    return common_images

            
def get_aruco_pose(image_name):
    for line in aruco_lines:
        if line.split(' ')[0].strip() == image_name:
            tvec = np.array([float(x) for x in line.split(' ')[1:4]])
            rvec = np.array([float(x) for x in line.split(' ')[4:7]])
            return tvec, rvec
    
def get_pnp_pose(image_name):
    for line in pnp_lines:
        if line.split(' ')[0].strip() == image_name:
            tvec = np.array([float(x) for x in line.split(' ')[1:4]])
            rvec = np.array([float(x) for x in line.split(' ')[4:7]])
            return tvec, rvec



def computeMetrics(aruco_t_vec,aruco_r_vec, pnp_t_vec,pnp_r_vec):
    
    dist_error = np.linalg.norm(aruco_t_vec - pnp_t_vec)
    est_dist = np.linalg.norm(aruco_t_vec)
    
    return [dist_error,est_dist]



metric_list = None
for image_name in extract_common_images():
    aruco_t_vec, aruco_r_vec = get_aruco_pose(image_name)
    pnp_t_vec, pnp_r_vec     = get_pnp_pose(image_name)
    
    
    if SHOW_AXIS:
        print(IMAGES_PATH+image_name)
        image = cv2.imread(IMAGES_PATH + image_name)
        aruco_axis = aruco.drawAxis(image, cameraMatrix, distCoeffs, aruco_r_vec, aruco_t_vec, 0.1)
        pnp_axis = aruco.drawAxis(aruco_axis, cameraMatrix, distCoeffs, pnp_r_vec, pnp_t_vec, 0.3)
        
        # cv2.imshow('aruco_axis', aruco_axis)
        cv2.imshow('pnp_axis', pnp_axis)
        cv2.waitKey(0) 
        
    metrics = computeMetrics(aruco_t_vec,aruco_r_vec, pnp_t_vec,pnp_r_vec)
    
    if metric_list is None:
        metric_list = np.array([metrics])
    else:
        metric_list = np.concatenate((metric_list, np.array([metrics])), axis=0)
        
    
    

if SHOW_AXIS:
    cv2.destroyAllWindows()


STEP = 1

def compute_total_metrics():
    mean_error =np.mean(metric_list[:,0])
    var_error =np.var(metric_list[:,0])
    max_dist = int(np.max(metric_list[:,1]))
    max_dist += max_dist % STEP 
        
    metrics_per_range = []
    
    for i in range(max_dist//STEP):
        temp_metric = []
        for index,elem in enumerate(metric_list[:,1]):
            if elem >= i*STEP and elem <= (i+1)*STEP:
                temp_metric.append(metric_list[index,0])
        
        metrics_per_range.append(temp_metric)
            
    print('mean_error:',mean_error)
    print('var_error:',var_error)
    print('max_dist:',max_dist)
    print('metrics_per_range',len(metrics_per_range))
    return [mean_error,var_error,max_dist,metrics_per_range]

mean_error,var_error,max_dist,metrics_per_range = compute_total_metrics()

# DRAW PLOT

import numpy as np
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(metrics_per_range)
ax1.set_yticks(np.arange(0, max_dist, STEP))
plt.show()



    
