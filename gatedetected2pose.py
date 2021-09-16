from PlotUtils import showLabels
from PAF import getGates, getSides
from PoseEstimation import estimateGatePose, getCameraParams
from GateNet import TrainableGateNet
from createTags import image2net, plotGates
import torch, os, time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    cameraMatrix = np.array([[790.8989920514197, 0.0, 670.332791421756],[0.0, 789.6808338497912, 370.6481124492188], [0.0, 0.0, 1.0]])
    distCoeffs   = np.array([-0.03448682771417174, -0.055932650937412745, 0.11969799783448262, -0.09163586323944228])

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    # device="cpu"

    PATH_FILE = "./estimation_net.txt"

    PATH_IMAGES = "./Dataset/Data_Adam/gates"
    image_dims = (480,368)

    image_list = os.listdir(PATH_IMAGES)
    dataset = image_list

    # Load Model
    CHECKPOINT_PATH = "checkpoints/September-10-2021_11_39AM_GateNet_11.pth"

    net = TrainableGateNet('PAFGauss')
    net.load_state_dict(torch.load(CHECKPOINT_PATH))
    net = net.to(device)

    avg = 0
    for i in tqdm(range(len(dataset))):
        i = 87
        # TEST IMAGES
        image_name = dataset[i]
        print(image_name)

        image = image2net(image_name, PATH_IMAGES, image_dims)

        net_input = torch.unsqueeze(image,0).to(device)
        
        start_time = time.time()
        output = net(net_input)
        end_time = time.time()
        dif_time = end_time-start_time
        avg = 0.1 * dif_time + 0.9 * avg
        # print('time ',dif_time,'avg',avg ,' freq:',1/(avg))

        labels=output.detach().to('cpu')[0].numpy()

        if type(image) == torch.Tensor:
            image = image.numpy() * 255
            image = np.array(image,dtype=np.uint8)
            image = np.transpose(image,(1,2,0))

        p = showLabels(image,labels)

        detected_sides = getSides(labels)
        # print(detected_sides)
        detected_gates = getGates(detected_sides)
        _, gate_poses = estimateGatePose(detected_gates, cameraMatrix,distCoeffs)

        lines=[]
        for gate in gate_poses:
            rvecs = gate[0]
            tvecs = gate[1]
            line = f'{image_name} '
            for data in tvecs:
                line = line + f'{data[0]:.6f} '
            for data in rvecs:
                line = line + f'{data[0]:.6f} '
            line = line[:-1] + '\n'
            lines.append(line)

        with open(PATH_FILE,'a') as f:
            for line in lines:
                f.write(line)




