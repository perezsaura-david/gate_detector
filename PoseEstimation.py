import numpy as np
import cv2, time, glob
from tqdm import tqdm

def getCameraParams():

    print('Get camera parameters')

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    grid_shape = [7,9]
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_shape[0]*grid_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_shape[0],0:grid_shape[1]].T.reshape(-1,2) * 0.019 # Square size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./Cam_Calibration/*.JPG')

    images_found = []

    time_0 = time.time()

    image_dims = (240,192)
    image_dims = (480,384)
    print('Finding images')

    for fname in tqdm(images):
        img = cv2.imread(fname)
        img = cv2.resize(img, image_dims, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgshow = gray #cv2.resize(gray, (960, 540)) 
        # cv2.imshow('img', imgshow)
        # print('Press ANY key to continue')
        # cv2.waitKey()
        # Find the chess board corners
        # print('Finding corners')
        ret, corners = cv2.findChessboardCorners(gray, (grid_shape[0],grid_shape[1]), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        # print('Done')
        # If found, add object points, image points (after refining them)
        if ret == True:
            # print('Corners found')
            objpoints.append(objp)
            # print('Refining points')
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # print('Done')
            imgpoints.append(corners)
            # Draw and display the corners
            # print('Drawing corners')
            cv2.drawChessboardCorners(img, (grid_shape[0],grid_shape[1]), corners2, ret)
            # print('Done')
            images_found.append(fname)
            # imgshow = img #cv2.resize(img, (960, 540)) 
            cv2.imshow('img', img)
            print('Press ANY key to continue')
            cv2.waitKey()
        # else:
            # print('Corners not found')
    print('Time elapsed finding images:', time.time() - time_0)
    # cv2.destroyAllWindows()
    print('Images found:', len(images_found),len(images))

    print('Calibrating camera')
    time_0 = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Time elapsed calibrating camera:', time.time() - time_0)
    # print('ret',ret)
    # print('matrix',mtx.shape, mtx)
    # print('distorsion',dist.shape, dist)
    # print('rvecs',np.shape(rvecs), rvecs)
    # print('tvecs',np.shape(tvecs), tvecs)

    return mtx

def estimateGatePose(detected_gates, camera_matrix):

    gate_poses = []

    for gate in detected_gates:

        gate_corners = np.array([gate['c0'][::-1],gate['c1'][::-1],gate['c2'][::-1],gate['c3'][::-1]], dtype=np.float32)
        gate_pose = estimatePose(gate_corners, camera_matrix)
        gate_poses.append(gate_pose)


    return gate_poses

def estimatePose(image_points, camera_matrix):
    # print(image_points)

    # gate_size = 2.4384 # meters
    # object_points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]) * gate_size
    gate_size = 1.2192 # meters
    object_points = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]) * gate_size

    # camera_matrix = cameraCalibration()
    dist_coeffs = np.zeros((1,5))
    success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,imagePoints=image_points,cameraMatrix=camera_matrix,distCoeffs=dist_coeffs,)

    return translation_vector



if __name__ == "__main__":

    camera_matrix = getCameraParams()
    # dist_coeffs = None
    # object_points = 
    # image_points =
    # success, rotation_vector, translation_vector = cv2.solvePnP(objectPoints=object_points,imagePoints=image_points,cameraMatrix=camera_matrix,distCoeffs=dist_coeffs)