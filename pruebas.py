# from PAF import integratePathBtwCorners

import cv2, torch
import numpy as np
from tqdm import tqdm

from createTags import PAFDataset
from PlotUtils import showLabels
from utils import normalizeLabels, resizeLabels
from PAF import getGates




if __name__ == "__main__":

    # # Usar coordenadas de una images: origen en esquina superior izquierda
    # th_dist = 5
    # # img_size = [240,480]
    # image_dims = [50,100]
    # # img_size = [4,8]

    # # gate1 = [[55,55],[155,415]]
    # # gate2 = [[205,305],[15,155]]
    # # gate3 = [[15,25],[35,450]]

    # gate1 = [[45,5],[15,81]]
    # gate2 = [[40,60],[20,10]]
    # gate3 = [[1,2],[40,90]]

    # gates_corners = [gate1,gate2,gate3]

    # gates_corners = [[[25,50],[25,0]],
    #                  [[25,50],[0,0]],
    #                  [[25,50],[0,50]],
    #                  [[25,50],[0,100]],
    #                  [[25,50],[25,100]],
    #                  [[25,50],[50,100]],
    #                  [[25,50],[50,50]],
    #                  [[25,50],[50,0]]]

    # gate1 = [[10,20],[40,80]]
    # # gate1 = [[1,2],[3,3]]
    # gates_corners = [gate1]

    # side_gates = np.array(gates_corners)

    PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    PATH_IMAGES  = "./Dataset/Data_Training/"
    image_dims = (480,360)

    dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

    for i in tqdm(range(len(dataset))):

        image, labels = dataset[i]

        if type(image) == torch.Tensor:
            # print(image.shape)
            image = image.numpy() * 255
            image = np.array(image,dtype=np.uint8)
            image = np.transpose(image,(1,2,0))

        labels = labels.detach().numpy()

        p = showLabels(image, labels)

        if p == 27 or p == ord('q'):
            break
        else:
            continue

exit()


if len(points) > 0:
    plt.scatter(points[0,:,0],points[0,:,1], c='r')
if len(points) > 1:
    plt.scatter(points[1,:,0],points[1,:,1], c='g')
if len(points) > 2:
    plt.scatter(points[2,:,0],points[2,:,1], c='b')
if len(points) > 3:
    plt.scatter(points[3,:,0],points[3,:,1], c='y')

if len(lines) > 0:
    plt.plot(lines[0,:,0,0],lines[0,:,0,1], c='r')
if len(lines) > 1:
    plt.plot(lines[1,:,0,0],lines[1,:,0,1], c='g')
if len(lines) > 2:
    plt.plot(lines[2,:,0,0],lines[2,:,0,1], c='b')
if len(lines) > 3:
    plt.plot(lines[3,:,0,0],lines[3,:,0,1], c='y')

# Tamaño de celda por pixel. -> Done
# Representar con HSV -> Done
# LineIterator OpenCV Iterar -> Done

# integratePathBtwCorners
def integrateSidePath(corners, vx_map_sum, vy_map_sum):

    dist_subpoints = 1
    _ , v_points = corners2Vector(corners, dist_subpoints)
    v_points, v_plot = points2grid(v_points, grid_size, c_grid_plot)

    v_idx_map = np.zeros_like(vx_map_sum.transpose())


    for i in range(len(v_idx_map)):
        for j in range(len(v_idx_map[i])):
            # Limited by the corner points
            if (i < min(v_points[:,0]) or i > max(v_points[:,0])):
                continue
            if (j < min(v_points[:,1]) or j > max(v_points[:,1])):
                continue
            for x,y in v_points:
                dist = np.sqrt((i-x)**2+(j-y)**2)
                if dist < 1:
                    v_idx_map[i,j] = 1

    v_idx = v_idx_map.transpose()

    score_x = sum(vx_map_sum[v_idx>0])
    score_y = sum(vy_map_sum[v_idx>0])
    print(score_x, score_y)
    # for x,y in v_points:
    #     v_idx_map[x,y] = 1

    return v_idx, score_x, score_y

corners_detected = []

for side in gates_corners:
    for corner in side:
        corners_detected.append(corner)

# random.shuffle(corners_detected)
# print(corners_detected)

# for side_gate in side_gates:

#     corner_score = np.zeros(corners_detected)

corners_detected = np.array(corners_detected)

print(corners_detected[:2])

test_corners = corners_detected[:2]

# corner0 = corners_detected[0] // grid_size
# corner1 = corners_detected[1] // grid_size

# corners = np.array(corner0,corner1)

v_idx, _, _ = integrateSidePath(test_corners,vx_map_sum, vy_map_sum)

vx_map_sum = v_idx

plotVecMaps(img_size, grid_plot, c_grid_plot, vx_map_sum, vy_map_sum, side_gates, v_points_plot)

# for gate in gates_corners:

#     print(gate)
    
#     for 
#     corners_grid = point2grid()
    
    # path_score = integratePathBtwCorners(xy_corners, vx_map_sum, vy_map_sum)

### Corner distribution ###
# PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
# PATH_IMAGES  = "./Dataset/Data_Training/"
# image_dims = (480,360)

# dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

# for i in tqdm(range(len(dataset)//10)):
#     data = dataset[i]
#     if len(data) > 0:
#         plt.scatter(data[0,:,0],data[0,:,1], c='r')
#     if len(data) > 1:
#         plt.scatter(data[1,:,0],data[1,:,1], c='g')
#     if len(data) > 2:
#         plt.scatter(data[2,:,0],data[2,:,1], c='b')
#     if len(data) > 3:
#         plt.scatter(data[3,:,0],data[3,:,1], c='y')
# plt.show()