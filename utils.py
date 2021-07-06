import numpy as np
import cv2
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

    # print(heigth,width)
    gaussMapBinary = np.ones(([heigth, width]),dtype=np.uint8)
    gaussMap = np.zeros([heigth, width])#,dtype=np.uint8)
    
    # print(gaussMap.shape)
    for label in labels:
        try:
            x1,y1,x2,y2,x3,y3,x4,y4 = label
            coordinates = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) * (width,heigth)
            for x,y in coordinates:
                gaussMapBinary[int(y)][int(x)] = 0

        except:
            print('empty label')

    gaussMap = cv2.distanceTransform(gaussMapBinary, cv2.DIST_L2, 0)
    gaussMap = np.exp(np.divide(-gaussMap,2*sigma*sigma))


    return gaussMap


if __name__ == "__main__":

    dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)

    for image,label in dataset:
        mapa = MakeGaussMap(image,label)
        cv2.imshow('map',mapa)
        cv2.waitKey()