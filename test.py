import  os, cv2,time

import numpy
from GateNet import *
from createTags import *
import torch

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
device="cpu"

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)
# image_dims = (240,180)
# image_dims = (480,100)


CHECKPOINT_PATH = "./checkpoints/July-08-2021_20_18PM_GateNet_7.pth"

net = TrainableGateNet('PAFGauss')
net.load_state_dict(torch.load(CHECKPOINT_PATH))

net = net.to(device)

dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

avg = 0
for image,label in dataset:
    
    start_time = time.time()
    output = net(torch.unsqueeze(image,0).to(device))
    output=output.detach().to('cpu')[0]
    print(output.shape)
    out = torch.zeros((1,output.shape[1],output.shape[2]))
    map =None
    for j, map in enumerate(output):
        out[0] += map
        cv2.imshow('label'+str(j), map.numpy()*5)
    # plotGates(image,out,'Gaussian',show = True)
    plotGates(image,out,'Gaussian',show = True)
    
   
    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',avg ,' freq:',1/(avg))
    