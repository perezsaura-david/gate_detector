import numpy as np
import torch, cv2

from skimage.draw import line
from skimage.feature import peak_local_max
from sklearn.metrics import mean_squared_error


def makeGaussMap(image,labels,scale_factor = 2,sigma=1.25):

    if type(image) == torch.Tensor:
        heigth = image.shape[1]
        width = image.shape[2]
    else:
        heigth = image.shape[0]
        width = image.shape[1]
    
    heigth = heigth // scale_factor
    width = width // scale_factor

    gauss_map_binary = np.ones(([heigth, width]),dtype=np.uint8)
    gauss_map = np.zeros([heigth, width])#,dtype=np.uint8)
    
    # Get labels
    for x,y in labels:
        gauss_map_binary[int(y * heigth)][int(x * width)] = 0

    gauss_map = cv2.distanceTransform(gauss_map_binary, cv2.DIST_L2, 0)
    gauss_map = np.exp(np.divide(-gauss_map,2*sigma*sigma))

    return gauss_map


# Normalize labels between (0,1)
def normalizeLabels(label_list, original_width, original_height):

    normalized_labels = []

    for label in label_list:
        normalized_label = np.zeros_like(label, dtype=np.float64)
        normalized_label[:,0] = label[:,0] / original_width
        normalized_label[:,1] = label[:,1] / original_height
        normalized_labels.append(normalized_label)

    # normalized_labels = np.array(normalized_labels)

    return normalized_labels


# Resize labels from normalized labels
def resizeLabels(label_list, new_width, new_height):

    resized_labels = []

    for label in label_list:
        resized_label = np.zeros_like(label, dtype=np.int32)
        resized_label[:,0] = label[:,0] * new_width
        resized_label[:,1] = label[:,1] * new_height
        resized_labels.append(resized_label)

    # normalized_labels = np.array(normalized_labels)

    return resized_labels

# For non-grouped labels
# def normalizeLabels(labels, original_width, original_height):
#     normalizedLabels = []
#     for label in labels:
#         normalizedLabel = []
#         for i, value in enumerate(label):
#             if i % 2 == 0:
#                 value = value / original_width
#             else:
#                 value = value / original_height
#             normalizedLabel.append(value)
#         normalizedLabels.append(normalizedLabel)

#     normalizedLabels = np.array(normalizedLabels)

#     return normalizedLabels


def groupCorners(points):
    corner_list = []
    
    for i in range(len(points)):
        if i%2 == 0:
            x = points[i]
        else:
            y = points[i]
            coord = [x,y]
            corner_points = np.array([coord])
            corner_list.append(corner_points)
    
    # corner_array = np.array(corner_list)
    return corner_list


def corners2Vector(corner_0,corner_1):

    # Calculate vector between corner points
    vector = np.array([corner_1[0] - corner_0[0],corner_1[1] - corner_0[1]])
    vector_1 = vector / np.linalg.norm(vector)

    # Divide the line in points
    v_points = np.array(list(zip(*line(int(corner_0[0]),int(corner_0[1]), int(corner_1[0]),int(corner_1[1])))))

    return vector, vector_1, v_points


def cleanGaussianMap(gauss_map, g_th=0.5):

    if type(gauss_map) == torch.Tensor:
        gauss_map = gauss_map.detach().numpy()

    g_map = gauss_map.squeeze()
    g_map = g_map * (1/np.max(g_map)) # Rescale map
    g_map = g_map * (g_map > g_th)     # Get > 0.5

    return g_map


def getCornersFromGaussMap(corner_maps):

    corners_detected = []
    for gauss_map in corner_maps:

        g_map = cleanGaussianMap(gauss_map)

        coordinates = peak_local_max(g_map, min_distance=6)
        corners_detected.append(coordinates)

    return corners_detected


def integratePathBtwCorners(corner_0, corner_1, vx_map, vy_map):

    # img0 = plotPAFimg(vx_map,vy_map)
    # cv2.imshow('Paf map',img0)

    # vx_map_plot = vx_mapq
    # vy_map_plot = vy_map

    vector, _, v_points = corners2Vector(corner_0,corner_1)
    
    path_vector_x = 0
    path_vector_y = 0
    for x,y in v_points:
        path_vector_x += vx_map[x,y]
        path_vector_y += vy_map[x,y]
        # vx_map_plot[x,y] = 1
        # vy_map_plot[x,y] = 1
    
    path_vector = np.array([path_vector_y,path_vector_x])
    # path_vector_1 = path_vector / np.linalg.norm(path_vector)

    # img1 = plotPAFimg(vx_map_plot,vy_map_plot)
    # cv2.imshow('Integrated path',img1)
    # cv2.waitKey()

    score = mean_squared_error(vector, path_vector, squared=False)
    return score


def calculateAffinityBtwCorners(corner_0, corner_1, vx_map, vy_map):

    _, vector_1, v_points = corners2Vector(corner_0,corner_1)
    
    score_list = []
    for x,y in v_points:
        paf_vector = [vy_map[x,y],vx_map[x,y]]
        paf_vnorm = np.linalg.norm(paf_vector)
        if paf_vnorm == 0:
            cosine = 0
        else:
            paf_vector_1 = paf_vector / paf_vnorm
            cosine = np.dot(vector_1, paf_vector_1)
        # angle = np.arccos(cosine)
        score_list.append(cosine)

    score = sum(score_list)/len(score_list)
    return score


def getCandidateSides(corners, i, j):

    side_list = []

    for ci_point in corners[i]:
        for cj_point in corners[j]:
            side_list.append(np.array([ci_point, cj_point]))

    return side_list


def deleteCandidates(c_list, score_list, idx):

    point_i = c_list[idx][0]
    point_j = c_list[idx][1]

    new_c_list = []
    new_score_list = []
    for i in range(len(c_list)):
        if (point_i in c_list[i]) or (point_j in c_list[i]): # Check this condition because we are comparing arrays
            continue
        else:
            new_c_list.append(c_list[i])
            new_score_list.append(score_list[i])

    return new_c_list, new_score_list


def dict2arrayGates(dict_list):

    gates_list = []
    for dic in dict_list:
        gate = []
        for i in range(5):
            c_name = 'c'+str(i)
            point = dic[c_name]
            if point is None:
                point = np.array([-1,-1])
            gate.append(point)
        gates_list.append(gate)

    gates_array = np.array(gates_list)

    return gates_array

# if __name__ == "__main__":

#     dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)

#     for image,label in dataset:
#         mapa = MakeGaussMap(image,label)
#         cv2.imshow('map',mapa)
#         k = cv2.waitKey()
#         if k == 27:
#             break