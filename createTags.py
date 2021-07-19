import torch, cv2, json
import numpy as np
from utils import * 
import pdb

from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from skimage import data, img_as_float



class gatesDataset(torch.utils.data.Dataset):
    """
    AlphaPilot torch dataset object. 
    Returns an image and a label's list
    """

    def __init__ (self,image_dims, path_images, path_labels, label_transformations=None ):
        super(gatesDataset).__init__()

        self.width = image_dims[0]
        self.height = image_dims[1]

        self.path_images = path_images
        self.path_labels = path_labels
        self.label_transformations = label_transformations
        
        self.filenames = []
        self.labelsDict = None

        with open(path_labels,'r') as json_file:
            self.labelsDict = json.load(json_file)

        for key in self.labelsDict.keys():
            self.filenames.append(key)

        sorted(self.filenames)

    def __len__(self):
        
        return (len(self.filenames))

    def __getitem__(self,key):
        image_filename = self.filenames[key]
        labels = self.labelsDict[image_filename]
        image = cv2.imread(self.path_images+image_filename)

        original_height , original_width , _ = image.shape

        # Normalize image

        image = cv2.resize(image,(self.width,self.height),interpolation = cv2.INTER_AREA)
        image = image / 255
        
        # Normalize labels between (0,1)
        normalizedLabels = []
        for label in labels:
            normalizedLabel = []
            for i, value in enumerate(label):
                if i % 2 == 0:
                    value = value / original_width
                else:
                    value = value / original_height
                normalizedLabel.append(value)
            normalizedLabels.append(normalizedLabel)

        normalizedLabels = np.array(normalizedLabels)

        # print(normalizedLabels)
        if self.label_transformations is None:
            if normalizedLabels.shape[1] != 8:
                normalizedLabels = np.zeros([1, 8])

        elif self.label_transformations == 'AddObjectness':
            if normalizedLabels.shape[1] != 8:
                normalizedLabels = np.zeros([1, 9])
            else:
                normalizedLabels =np.insert(normalizedLabels,0,1,1)

        elif self.label_transformations == 'Gaussian':
            try:
                x1,y1,x2,y2,x3,y3,x4,y4 = normalizedLabels[0]
                coordinates = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) 
            except:
                print("no labels")
                coordinates = np.array([])

            normalizedLabels = np.array(MakeGaussMap(image,coordinates))
            normalizedLabels = normalizedLabels.reshape((1,normalizedLabels.shape[0],normalizedLabels.shape[1]))
            


        elif self.label_transformations == 'PAFGauss':
            labelsList = []
            # List of corners (4)
            normalizedCorner = groupCorners(normalizedLabels[0])
            for i in range(len(normalizedCorner)):
                corners = normalizedCorner[i]
                normalizedLabels = np.array(MakeGaussMap(image,corners))
                normalizedLabels = normalizedLabels.reshape((1,normalizedLabels.shape[0],normalizedLabels.shape[1]))
                N_labels = torch.from_numpy(normalizedLabels)
                labelsList.append(N_labels[0])
        else:
            raise ValueError('Label_tranformation ' + str(self.label_transformations) +' does not exists')
        
        
        if len(normalizedLabels.shape) != 3:
            print('aqui entro')
            print(self.labelsDict[image_filename])
            print(image_filename)
            print(normalizedLabels)
            print(normalizedLabels.shape)
            
            
        N4_labels = torch.zeros((4,normalizedLabels.shape[1],normalizedLabels.shape[2]))

        for i in range(4):
            N4_labels[i] = labelsList[i]

        return torch.Tensor(np.transpose(image,(2,0,1))), N4_labels


ColorLists = [(255,0,0),(0,255,0),(0,0,255),(255,255,255),(0,0,0),(125,125,125)]

def plotGates(image,label,label_type = None,show = True):
    """
    Method for ploting gates estimation in an image
    """

    if type(image) == torch.Tensor:
        # print(image.shape)
        image = image.numpy() * 255
        image = np.array(image,dtype=np.uint8)
        image = np.transpose(image,(1,2,0))
    
    if type(label) == torch.Tensor:
        label = label.detach().numpy()

    if label_type is None:

        #  Check inputs
        if len(image.shape) != 3:
            raise AssertionError('Image vector is ' +str(len(image.shape))+'-dimensional, it must be 3-dimensional')
        if len(label.shape) != 2:
            raise AssertionError('Label vector is ' +str(len(label.shape))+'-dimensional, it must be 2-dimensional')
        
        # If inputs are torch.Tensor transform it into np.array


        image_h, image_w, image_ch = image.shape

        #check if there is any gate in image
        if label.any():
            #if there is some gates check label dimension
            if label.shape[1] == 9:
                label=np.delete(label,0,1)
            elif label.shape[1] != 8:
                raise ValueError('Label 2-nd component has '+str(label.shape[1]) +' elements, it must have 8: (x1,x2,x3,x4,x5,x6,x7,x8)')
            
            for i,gate_label in enumerate(label):
                x1,y1,x2,y2,x3,y3,x4,y4 = gate_label
                cnt = np.array(np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) * np.array([image_w,image_h]),dtype=np.int32)
                if i >= len(ColorLists):
                    raise KeyError('labeled gates number is larger than the length of ColorVector, increase the ColorVector list')
                image= cv2.UMat(image)
                cv2.drawContours(image, [cnt], -1, ColorLists[i], 3)


    elif label_type == 'Gaussian':
        label=label.squeeze()
        label = label * (1/np.max(label))

        # peaks, a = find_peaks(label[1])
        label = label * (label > 0.5)
        
        
        im = label
        coordinates = peak_local_max(im, min_distance=6)*2
        # print(coordinates)

        # print(coordinates.shape)
        if coordinates.shape[0] == 4:
            for x,y in coordinates:
                if x in sorted(coordinates[:,0])[0:2]:
                    if y in sorted(coordinates[:,1])[0:2]:
                        y1,x1 = x,y
                    else:
                        y2,x2 = x,y     
                else:
                    if y in sorted(coordinates[:,1])[0:2]:
                        y4,x4 = x,y
                    else:
                        y3,x3 = x,y
            cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            image= cv2.UMat(image)
            cv2.drawContours(image, [cnt], -1, (255,0,0), 2)

        else:
            print('only detected ' + str(coordinates.shape[0])+ 'corners')
            # cv2.imshow('Im label',im)
            # cv2.imshow('image Labeled',image)
            # cv2.waitKey()
            
        
            # cv2.imshow('GaussianMap',label)
            # pdb.set_trace()
        if show == True:    
            cv2.imshow('GaussianMap',label)
        
    if show == True:        
        cv2.imshow('image Labeled',image)
        cv2.waitKey()

