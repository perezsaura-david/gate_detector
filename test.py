import  os, cv2,time
from GateNet import *
from createTags import *
import torch

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)
# image_dims = (240,180)
# image_dims = (480,100)


CHECKPOINT_PATH = "./checkpoints/July-06-2021_17_01PM_GateNet_0.pth"

net = TrainableGateNet('Gaussian')
net.load_state_dict(torch.load(CHECKPOINT_PATH))

net = net.to(device)

dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='Gaussian')

avg = 0
for image,label in dataset:
    
    start_time = time.time()
    output = net(torch.unsqueeze(image,0).to(device))
    output=output.detach().to('cpu')
    plotGates(image,output,'Gaussian',show = True)

    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',avg ,' freq:',1/(avg))
    