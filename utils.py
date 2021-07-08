import numpy as np
import cv2
import torch
from  createTags import *

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)

def MakeGaussMap(image,labels,scale_factor = 2,sigma=1.25):

    if type(image) == torch.Tensor:
        heigth = image.shape[1]
        width = image.shape[2]
    else:
        heigth = image.shape[0]
        width = image.shape[1]
    
    heigth = heigth // scale_factor
    width = width // scale_factor

    gaussMapBinary = np.ones(([heigth, width]),dtype=np.uint8)
    gaussMap = np.zeros([heigth, width])#,dtype=np.uint8)
    
    # Get labels
    for x,y in labels:
        gaussMapBinary[int(y * heigth)][int(x * width)] = 0

    gaussMap = cv2.distanceTransform(gaussMapBinary, cv2.DIST_L2, 0)
    gaussMap = np.exp(np.divide(-gaussMap,2*sigma*sigma))


    return gaussMap

def groupCorners(points):
    cornerList = []
    for i in range(len(points)):
        if i%2 == 0:
            x = points[i]
        else:
            y = points[i]
            coord = [x,y]
            cornerPoints = [coord]
            cornerList.append(cornerPoints)
    return cornerList

# if __name__ == "__main__":

#     dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)

#     for image,label in dataset:
#         mapa = MakeGaussMap(image,label)
#         cv2.imshow('map',mapa)
#         k = cv2.waitKey()
#         if k == 27:
#             break