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


CHECKPOINT_PATH = "./checkpoints/July-31-2021_20_00PM_GateNet_213.pth"


net = TrainableGateNet('PAFGauss')
net.load_state_dict(torch.load(CHECKPOINT_PATH))

net = net.to(device)

dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

avg = 0
for image,label in dataset:
    
    start_time = time.time()
    output = net(torch.unsqueeze(image,0).to(device))
    output=output.detach().to('cpu')[0].numpy()
    print(output.shape)

    corners = output[:4]
    vx_map  = output[4:8]
    vy_map  = output[8:]

    gauss_map = torch.zeros((1,output.shape[1],output.shape[2]))
    map =None
    for j, map in enumerate(corners):
        gauss_map[0] += map
        cv2.imshow('label'+str(j), map*5)
    plotGates(image,gauss_map,'Gaussian',show = True)

    print("PAF")

    vx_map_sum = np.zeros_like(vx_map[0])
    for map in vx_map:
        vx_map_sum += map

    vy_map_sum = np.zeros_like(vy_map[0])
    for map in vy_map:
        vy_map_sum += map

    p = plotPAFimg(vx_map_sum,vy_map_sum)
    
    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',avg ,' freq:',1/(avg))

    if p == 27:
        break
    