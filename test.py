import torch, time
from tqdm import tqdm

from GateNet import *
from PlotUtils import showLabels

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
device="cpu"

# Load training dataset
PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,368)

# dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')


# # Load test dataset
# PATH_IMAGES = "/home/dps/rosbag_gates/images/one_gate/2021-07-09-10-12-46/"
PATH_IMAGES  = "./Dataset/Data_Test/"
image_list = os.listdir(PATH_IMAGES)
dataset = image_list

# Load Model
CHECKPOINT_PATH = "checkpoints/August-04-2021_09_53AM_GateNet_49.pth"

net = TrainableGateNet('PAFGauss')
net.load_state_dict(torch.load(CHECKPOINT_PATH))
net = net.to(device)

avg = 0
# for image,label in dataset:
for i in tqdm(range(len(dataset))):

    # TEST IMAGES
    # i = 9 # Test gate for PNP
    image_name = dataset[i]
    image = image2net(image_name, PATH_IMAGES, image_dims)

    # TRAINING IMAGES
    # image, _ = dataset[i]
    # image1, _ = dataset[i+1]
    # image += image1

    net_input = torch.unsqueeze(image,0).to(device)
    
    start_time = time.time()
    output = net(net_input)
    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',avg ,' freq:',1/(avg))

    output=output.detach().to('cpu')[0].numpy()

    if type(image) == torch.Tensor:
        # print(image.shape)
        image = image.numpy() * 255
        image = np.array(image,dtype=np.uint8)
        image = np.transpose(image,(1,2,0))

    p = showLabels(image,output)
    
    if p == 27 or p == ord('q'):
        break
    else:
        continue