from createTags import plotGates
from utils import getCornersFromGaussMap, normalizeLabels, resizeLabels
from PAF import getGates
import torch, cv2, math
import numpy as np

# Plot utils
def makePAFimg(vx_map_sum,vy_map_sum):
    # HSV
    # Hue        -> orientation
    # Saturation -> const 1
    # Brightness -> magnitude
    hsv_map = np.zeros((vx_map_sum.shape[0],vx_map_sum.shape[1],3), dtype='float32')
    for i in range(vx_map_sum.shape[0]):
        for j in range(vx_map_sum.shape[1]):
            hsv_map[i,j,0] = (math.atan2(vx_map_sum[i,j],vy_map_sum[i,j])+np.pi)*360/(2*np.pi) # Orientation
            hsv_map[i,j,1] = 1.0    # Saturation
            hsv_map[i,j,2] = np.linalg.norm([vx_map_sum[i,j],vy_map_sum[i,j]]) # Magnitude
    
    # Normalize vector magnitude channel
    # hsv_map[:,:,2] = hsv_map[:,:,2] / np.amax(hsv_map[:,:,2])
    # Show image of each channel
    # for i in range(3):
    #     print(np.amax(hsv_map[:,:,i]),np.amin(hsv_map[:,:,i]))

    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2RGB)
    # plt.imshow(vx_map_sum, cmap='hsv')
    # plt.imshow(hsv_map)
    # plt.show()

    return hsv_map

def addCorners2Image(image, corners):

    color = [(0, 0, 255),(0,255,255),(255,255,0),(255,0,0)]
    for i in range(4):
        for j in range(len(corners[i])):
            cv2.circle(image, corners[i][j][::-1], 5, color[i], -1)

    return image

def addGates2Image(image, gates):

    color = (0,0,255)

    cv2.polylines(image, gates, isClosed=False, color=color, thickness=2)

    return image

def showLabels(image, labels):

    ### IMAGE ###

    if type(image) == torch.Tensor:
        # print(image.shape)
        image = image.numpy() * 255
        image = np.array(image,dtype=np.uint8)
        image = np.transpose(image,(1,2,0))

    image_dims = (image.shape[0],image.shape[1])

    ### LABELS ###

    corners = labels[:4]
    vx_map  = labels[4:8]
    vy_map  = labels[8:]

    ### GAUSS MAP ###

    gauss_map = torch.zeros((1,labels.shape[1],labels.shape[2]))
    g_map = None
    for j, g_map in enumerate(corners):
        gauss_map[0] += g_map
        # cv2.imshow('label'+str(j), map*5)

    gauss_map_img, gauss_gate_img = plotGates(image,gauss_map,'Gaussian',show = True)

    ### PAF ###

    paf_imgs = []
    for i in range(4):
        img = makePAFimg(vx_map[i],vy_map[i])
        paf_imgs.append(img*5)
    
    vx_map_sum = np.zeros_like(vx_map[0])
    for map in vx_map:
        vx_map_sum += map
    vy_map_sum = np.zeros_like(vy_map[0])
    for map in vy_map:
        vy_map_sum += map

    paf_map_img = makePAFimg(vx_map_sum,vy_map_sum)

    ### CORNERS ###

    coord = getCornersFromGaussMap(corners)
    normalized_coords = normalizeLabels(coord, image.shape[0],image.shape[1])
    resized_coords = resizeLabels(normalized_coords, image.shape[0], image.shape[1])

    ### GATES ###

    detected_gates   = getGates(image_dims, labels)
    connected_gates  = detected_gates[:,:,::-1]
    normalized_gates = normalizeLabels(connected_gates, labels.shape[1],labels.shape[2])
    resized_gates    = resizeLabels(normalized_gates, image.shape[0],image.shape[1])

    ### SHOW IMAGES ###

    # Show labels
    # Corner labels
    gauss_top = cv2.hconcat((corners[0],corners[1]))
    gauss_bot = cv2.hconcat((corners[3],corners[2]))
    gauss_corners = cv2.vconcat((gauss_top,gauss_bot))
    gauss_corners = cv2.cvtColor(gauss_corners*5, cv2.COLOR_GRAY2BGR)
    # PAF labels
    paf_top = cv2.hconcat((paf_imgs[0],paf_imgs[1]))
    paf_bot = cv2.hconcat((paf_imgs[3],paf_imgs[2]))
    paf_sides = cv2.vconcat((paf_top,paf_bot))
    labels_img = cv2.hconcat((gauss_corners,paf_sides))

    cv2.imshow('Corners & PAF labels', labels_img)

    # Show results
    gauss_map_img   = cv2.cvtColor(gauss_map_img, cv2.COLOR_GRAY2BGR)
    results_net_img = cv2.vconcat((gauss_map_img, paf_map_img))

    cv2.imshow('Results',results_net_img)

    image = addGates2Image(image, resized_gates)
    image = addCorners2Image(image,resized_coords)
    cv2.imshow('Detections', image)

    k = cv2.waitKey()

    return k