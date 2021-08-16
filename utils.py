import numpy as np
import torch, cv2

from skimage.draw import line
from skimage.feature import peak_local_max
from sklearn.metrics import mean_squared_error

# PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
# PATH_IMAGES  = "./Dataset/Data_Training/"
# image_dims = (480,360)

def MakeGaussMap(image,labels,scale_factor = 2,sigma=1.25):

    if type(image) == torch.Tensor:
        heigth = image.shape[1]
        width = image.shape[2]
    else:
        heigth = image.shape[0]
        width = image.shape[1]
    
    heigth = heigth // scale_factor
    width = width // scale_factor

    gaussMapBinary = np.ones(([heigth, width]),dtype=np.uint8)
    gaussMap = np.zeros([heigth, width])#,dtype=np.uint8)
    
    # Get labels
    for x,y in labels:
        gaussMapBinary[int(y * heigth)][int(x * width)] = 0

    gaussMap = cv2.distanceTransform(gaussMapBinary, cv2.DIST_L2, 0)
    gaussMap = np.exp(np.divide(-gaussMap,2*sigma*sigma))

    return gaussMap

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

def getCornersFromGaussMap(corner_maps):

    # gauss_map = np.zeros((1,corner_maps.shape[1],corner_maps.shape[2]))
    # map =None
    # for j, map in enumerate(corners):
    #     gauss_map[0] += map

    corners_detected = []
    for gauss_map in corner_maps:

        g_map = gauss_map.squeeze()
        g_map = g_map * (1/np.max(g_map)) # Rescale map
        # peaks, a = find_peaks(label[1])
        g_map = g_map * (g_map > 0.5)   # Get > 0.5

        coordinates = peak_local_max(g_map, min_distance=6)*2 # Img size x2 -> TODO: Change 
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

# if __name__ == "__main__":

#     dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)

#     for image,label in dataset:
#         mapa = MakeGaussMap(image,label)
#         cv2.imshow('map',mapa)
#         k = cv2.waitKey()
#         if k == 27:
#             break