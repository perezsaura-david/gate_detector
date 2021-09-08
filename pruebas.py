
import torch
import numpy as np
from tqdm import tqdm

from createTags import PAFDataset
from PAF import detectGates
from PoseEstimation import estimateGatePose
from PoseEstimation import getCameraParams
from PlotUtils  import showLabels
from utils import getCornersFromGaussMap
from metrics import getDetectionMetrics



if __name__ == "__main__":

    PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    PATH_IMAGES  = "./Dataset/Data_Training/"
    image_dims = (480,360)

    dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

    # camera_matrix = getCameraParams()

    for i in tqdm(range(len(dataset))):

        image, labels = dataset[i]

        if type(image) == torch.Tensor:
            image = image.numpy() * 255
            image = np.array(image,dtype=np.uint8)
            image = np.transpose(image,(1,2,0))

        if type(labels) == torch.Tensor:
            labels = labels.detach().numpy()


        gauss_maps = labels[:4]
        corners    = getCornersFromGaussMap(gauss_maps)
        print(corners)

        


        getDetectionMetrics(corners)


        exit()


        detected_gates = detectGates(labels)
        print(detected_gates)
        # pose_estimations = estimateGatePose(detected_gates, camera_matrix)

        # print('Pose estimation',pose_estimations)

        p = showLabels(image, labels)

        if p == 27 or p == ord('q'):
            break
        else:
            continue

# Tamaño de celda por pixel. -> Done
# Representar con HSV -> Done
# LineIterator OpenCV Iterar -> Done
# Cambiar DataLoader para corregir etiquetas.