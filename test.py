import torch, time
from GateNet import *
from PlotUtils import showLabels

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
device="cpu"

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,368)
# image_dims = (240,180)
# image_dims = (480,100)

CHECKPOINT_PATH = "checkpoints/August-04-2021_09_53AM_GateNet_49.pth"

net = TrainableGateNet('PAFGauss')
net.load_state_dict(torch.load(CHECKPOINT_PATH))

net = net.to(device)

dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

avg = 0
for image,label in dataset:
    
    start_time = time.time()
    output = net(torch.unsqueeze(image,0).to(device))
    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',avg ,' freq:',1/(avg))

    output=output.detach().to('cpu')[0].numpy()

    p = showLabels(image,output)
    
    if p == 27 or p == ord('q'):
        break
    else:
        continue