from PAF import plotPAFimg
from createTags import plotGates
import torch, cv2
import numpy as np

# Plot utils

def showLabels(image, labels):

    corners = labels[:4]
    vx_map  = labels[4:8]
    vy_map  = labels[8:]

    ### Corners ###
    gauss_map = torch.zeros((1,labels.shape[1],labels.shape[2]))
    map =None
    for j, map in enumerate(corners):
        gauss_map[0] += map
        # cv2.imshow('label'+str(j), map*5)

    gauss_map_img, gate_img = plotGates(image,gauss_map,'Gaussian',show = True)
    # cv2.imshow('Gaussian map', gauss_map_img)
    # cv2.imshow('Gate detected', gate_img)

    ### PAF ###
    paf_imgs = []
    for i in range(4):
        img = plotPAFimg(vx_map[i],vy_map[i])
        paf_imgs.append(img*5)
    
    vx_map_sum = np.zeros_like(vx_map[0])
    for map in vx_map:
        vx_map_sum += map
    vy_map_sum = np.zeros_like(vy_map[0])
    for map in vy_map:
        vy_map_sum += map

    paf_map_img = plotPAFimg(vx_map_sum,vy_map_sum)
    # cv2.imshow('PAF complete', paf_map_img)

    # Show labels
    # Corner labels
    gauss_top = cv2.hconcat((corners[0],corners[1]))
    gauss_bot = cv2.hconcat((corners[3],corners[2]))
    gauss_corners = cv2.vconcat((gauss_top,gauss_bot))
    gauss_corners = cv2.cvtColor(gauss_corners*5, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('Corners', gauss_corners)
    # PAF labels
    paf_top = cv2.hconcat((paf_imgs[0],paf_imgs[1]))
    paf_bot = cv2.hconcat((paf_imgs[2],paf_imgs[3]))
    paf_sides = cv2.vconcat((paf_top,paf_bot))
    # cv2.imshow('PAF sides', paf_sides)
    labels_img = cv2.hconcat((gauss_corners,paf_sides))
    cv2.imshow('Corners & PAF labels', labels_img)

    # Show results
    gauss_map_img   = cv2.cvtColor(gauss_map_img, cv2.COLOR_GRAY2BGR)
    results_net_img = cv2.vconcat((gauss_map_img, paf_map_img))
    # print(results_net_img.shape, gate_img.shape)
    # results_img = cv2.hconcat((results_net_img, gate_img))
    cv2.imshow('Results',results_net_img)
    cv2.imshow('Gate', gate_img)

    k = cv2.waitKey()

    return k