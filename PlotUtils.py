import torch, cv2, math
import numpy as np

from PAF   import getGates, getSides
from utils import cleanGaussianMap, getCornersFromGaussMap, normalizeLabels, resizeLabels, dict2arrayGates
from PoseEstimation import estimateGatePose

# FUNCTIONS
#   makePAFimg
#   addCorners2Image
#   addGates2Image
#   gates2plot
#   showLabels
#   plotVecMaps


# Plot utils
def makePAFimg(vx_map_sum,vy_map_sum):
    # HSV
    # Hue        -> Orientation
    # Saturation -> Const
    # Value      -> Magnitude
    hsv_map = np.zeros((vx_map_sum.shape[0],vx_map_sum.shape[1],3), dtype='float32')
    for i in range(vx_map_sum.shape[0]):
        for j in range(vx_map_sum.shape[1]):
            hsv_map[i,j,0] = (math.atan2(vx_map_sum[i,j],vy_map_sum[i,j])+np.pi)*360/(2*np.pi) # Orientation
            hsv_map[i,j,1] = 1.0    # Saturation
            hsv_map[i,j,2] = np.linalg.norm([vx_map_sum[i,j],vy_map_sum[i,j]]) # Magnitude

    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2RGB)

    return hsv_map

def addCorners2Image(image, corners):

    color = [(0, 0, 255),(0,255,255),(255,255,0),(255,0,0)]

    for i in range(4):
        for j in range(len(corners[i])):
            cv2.circle(image, corners[i][j][::-1], 5, color[i], -1)

    return image

def addDetections2Image(image, corners, gt, size=3):

    if gt:
        color = (0,0,255) # gt
        th = -1
        size = 5
    else:
        color = (0,255,0) # detections
        th = -1
        size = 3


    for i in range(4):
        for j in range(len(corners[i])):
            cv2.circle(image, corners[i][j][::-1], size, color, th)

    return image

def addGates2Image(image, gates):

    resized_gates  = gates2plot(gates)

    color = (0,255,0)
    # color = (0,0,0)

    for gate in resized_gates:
        for i in range(4):
            j = (i+1) % 4
            if any(gate[i] < 0) or any(gate[j] < 0):
                continue
            cv2.line(image, gate[i], gate[j], color=color, thickness=10)

    return image

def addSides2Image(image, sides):

    color = [(0, 0, 255),(0,255,255),(255,255,0),(255,0,0)]

    for i in range(4):
        for connection in sides[i]:
            # print(connection)
            if connection['score'] < 0.9:
                continue
            line = connection['side'][:,::-1] * 2
            cv2.line(image, line[0], line[1], color=color[i], thickness=5)

    return image

def gates2plot(gates_dict):

    gates_array = dict2arrayGates(gates_dict)
    gates_array = gates_array[:,:,::-1]

    resized_gates = np.zeros_like(gates_array, dtype=np.int32)

    resized_gates[:,:,0] = gates_array[:,:,0] * 2
    resized_gates[:,:,1] = gates_array[:,:,1] * 2

    return resized_gates

def addImgCenter2Image(image, camera_matrix):

    center_point = (int(camera_matrix[0,2]*2),int(camera_matrix[1,2]*2)) # (116*2,98*2)

    cv2.circle(image, center_point, 10, (0,255,0), -1) # AddCenter2Image

    return image



def showLabels(image, labels):

    ### LABELS ###

    corners = labels[:4]
    vx_map  = labels[4:8]
    vy_map  = labels[8:]

    ### GAUSS MAP ###

    gauss_map = torch.zeros((1,labels.shape[1],labels.shape[2])) # Why torch instead np.array?
    # gauss_map = np.zeros((1,labels.shape[1],labels.shape[2]))

    for g_map in corners:
        gauss_map[0] += g_map

    gauss_map_img = cleanGaussianMap(gauss_map)

    ### PAF ###

    emphasize = 1
    paf_imgs = []
    for i in range(4):
        img = makePAFimg(vx_map[i],vy_map[i])
        paf_imgs.append(img*emphasize)
    
    vx_map_sum = np.zeros_like(vx_map[0])
    for x_map in vx_map:
        vx_map_sum += x_map
    vy_map_sum = np.zeros_like(vy_map[0])
    for y_map in vy_map:
        vy_map_sum += y_map

    paf_map_img = makePAFimg(vx_map_sum,vy_map_sum)

    ### CORNERS ###

    coord             = getCornersFromGaussMap(corners)
    normalized_coords = normalizeLabels(coord, labels.shape[1],labels.shape[2])
    resized_coords    = resizeLabels(normalized_coords, image.shape[0], image.shape[1])

    ### GATES ###
    detected_sides = getSides(labels)
    detected_gates = getGates(detected_sides)

    ### Estimate pose ###
    # x = estimateGatePose(detected_gates)

    ### SHOW IMAGES ###
    # gm = []
    # gm_0 = cv2.cvtColor(corners[0], cv2.COLOR_GRAY2BGR)
    # gm_0[:,:,:2] *= 0
    # gm.append(gm_0)
    # gm_1 = cv2.cvtColor(corners[1], cv2.COLOR_GRAY2BGR)
    # gm_1[:,:,0] *= 0
    # gm.append(gm_1)
    # gm_2 = cv2.cvtColor(corners[2], cv2.COLOR_GRAY2BGR)
    # gm_2[:,:,2] *= 0
    # gm.append(gm_2)
    # gm_3 = cv2.cvtColor(corners[3], cv2.COLOR_GRAY2BGR)
    # gm_3[:,:,1:] *= 0
    # gm.append(gm_3)

    gm = corners


    # Add border to individual images
    border_width = 2
    border_color = (255,255,255)
    border_corner_imgs = []
    emphasize = 1
    for c_img in gm:
        border_img = cv2.copyMakeBorder(c_img*emphasize, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
        border_corner_imgs.append(border_img)

    border_paf_imgs = []
    for paf_img in paf_imgs:
        border_img = cv2.copyMakeBorder(paf_img, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
        border_paf_imgs.append(border_img)
    paf_imgs = border_paf_imgs


    # Show labels
    # Corner labels
    # gm_0 = corners[0]
    # gm_1 = corners[1]
    # gm_2 = corners[2]
    # gm_3 = corners[3]

    gauss_top = cv2.hconcat((border_corner_imgs[0],border_corner_imgs[1]))
    gauss_bot = cv2.hconcat((border_corner_imgs[3],border_corner_imgs[2]))
    gauss_corners_img = cv2.vconcat((gauss_top,gauss_bot))
    gauss_corners_img = cv2.cvtColor(gauss_corners_img, cv2.COLOR_GRAY2BGR)
    # PAF labels
    paf_top = cv2.hconcat((paf_imgs[0],paf_imgs[1]))
    paf_bot = cv2.hconcat((paf_imgs[3],paf_imgs[2]))
    paf_sides_img = cv2.vconcat((paf_top,paf_bot))

    # Add border to groups of images
    border_width = 2
    border_color = (255,255,255)
    b_gauss_corners_img = cv2.copyMakeBorder(gauss_corners_img, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
    b_paf_sides_img = cv2.copyMakeBorder(paf_sides_img, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
    
    labels_img = cv2.hconcat((b_gauss_corners_img,b_paf_sides_img))
    cv2.imshow('Corners & PAF labels', labels_img)

    # Show results
    gauss_map_img   = cv2.cvtColor(gauss_map_img, cv2.COLOR_GRAY2BGR)

    # Add border to individual images
    border_width = 1
    border_color = (255,255,255)
    b_gauss_map_img = cv2.copyMakeBorder(gauss_map_img, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
    b_paf_map_img   = cv2.copyMakeBorder(paf_map_img, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=border_color)
    
    results_net_img = cv2.vconcat((b_gauss_map_img, b_paf_map_img))

    cv2.imshow('Results',results_net_img)

    # image = addGates2Image(image, detected_gates)
    # image = addCorners2Image(image, resized_coords)
    # image = addImgCenter2Image(image, camera_matrix)
    image_gate = image.copy()
    image_gate = addGates2Image(image_gate, detected_gates)
    imge_side = image.copy()
    image = addSides2Image(image, detected_sides)
    image = addCorners2Image(image, resized_coords)


    
    cv2.imshow('gate',image_gate)
    cv2.imshow('Detections', image)

    # cv2.imshow('Corners', b_gauss_corners_img)
    # cv2.imshow('Corners gate', b_gauss_map_img)
    # cv2.imshow('PAF sides', b_paf_sides_img)
    # cv2.imshow('PAF', b_paf_map_img)

    k = cv2.waitKey()

    return k

def addAxis2Image(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

