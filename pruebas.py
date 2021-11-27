
import cv2
import torch
import numpy as np
from tqdm import tqdm

from createTags import PAFDataset
from PAF import detectGates
from PoseEstimation import estimateGatePose, projectAxis
from PoseEstimation import getCameraParams
from PlotUtils  import addAxis2Image, showLabels
from utils import getCornersFromGaussMap
from metrics import getDetectionMetrics



if __name__ == "__main__":

    PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    PATH_IMAGES  = "./Dataset/Data_Training/"
    image_dims = (480,360)

    dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

    # camera_matrix, distorsion = getCameraParams()

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

        # getDetectionMetrics(corners)

        detected_gates = detectGates(labels)




        print(detected_gates)
        # gate_corners, gate_estimations = estimateGatePose(detected_gates, camera_matrix)

        for i in range(len(gate_corners)):

            print('Gate estimation',gate_estimations[i])

            gate_points = gate_corners[i]
            gate_rot = gate_estimations[i][0]
            gate_pos = gate_estimations[i][1]

            imgpoints = projectAxis(gate_rot,gate_pos,camera_matrix,distorsion)
            image = addAxis2Image(image,gate_points, imgpoints)
        
        cv2.imshow('Ejes',image)
        p = cv2.waitKey()

        # p = showLabels(image, labels)

        if p == 27 or p == ord('q'):
            break
        else:
            continue

# Tamaño de celda por pixel. -> Done
# Representar con HSV -> Done
# LineIterator OpenCV Iterar -> Done
# Cambiar DataLoader para corregir etiquetas.