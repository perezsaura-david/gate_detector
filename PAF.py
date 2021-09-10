import torch
import numpy as np

from utils import calculateAffinityBtwCorners, corners2Vector, getCornersFromGaussMap, getCandidateSides, deleteCandidates

# FUNCTIONS: 
#   makeVecMaps
#   generatePAF
#   getSidesFromCorners
#   connectGatesFromSides
#   checkGates
#   getGates

# Make PAF map
def makeVecMaps(image_dims, corners, v_points, vector_1, th_dist):

    vx_map = np.zeros((image_dims[0],image_dims[1]))
    vy_map = np.zeros((image_dims[0],image_dims[1]))
    
    for i in range(len(vx_map)):
        for j in range(len(vx_map[i])):
            # Limited by the corner points
            if (i < (min(corners[:,0] - th_dist)) or i > (max(corners[:,0] + th_dist))):
                continue
            if (j < (min(corners[:,1] - th_dist)) or j > (max(corners[:,1] + th_dist))):
                continue
            
            for x,y in v_points:
                dist = np.sqrt((i-x)**2+(j-y)**2)
                if dist < th_dist:
                    if dist < 1:
                        value = 1
                    else:
                        value = 1/dist
                    vx_map[i,j] = max(value,vx_map[i,j])
                    vy_map[i,j] = max(value,vy_map[i,j])

    vx_map = vx_map * vector_1[0]
    vy_map = vy_map * vector_1[1]

    return vx_map, vy_map


def generatePAF(image, side_gates, scale_factor = 2, th_dist = 1):

    if type(image) == torch.Tensor:
        height = image.shape[1]
        width = image.shape[2]
    else:
        height  = image.shape[0]
        width = image.shape[1]

    height = height // scale_factor
    width  = width // scale_factor

    image_dims = (width,height)

    vx_map_sum = np.zeros(image_dims)
    vy_map_sum = np.zeros(image_dims)

    # If there is no corner detected, we return empty maps
    if len(side_gates) > 0:
        # v_points_plot = []
        for side_gate in side_gates:

            for corner in side_gate:
                corner[0] = int(round(corner[0] * width,0))
                corner[1] = int(round(corner[1] * height,0))

            _, vector_1, v_points = corners2Vector(side_gate[0],side_gate[1])

            vx_map, vy_map = makeVecMaps(image_dims, side_gate, v_points, vector_1, th_dist)

            vx_map_sum += vx_map
            vy_map_sum += vy_map

    vx_map_sum = vx_map_sum.transpose()
    vy_map_sum = vy_map_sum.transpose()

    return vx_map_sum, vy_map_sum


def getSidesFromCorners(corners, vx_maps, vy_maps, conf_th = 0.91):

    # Check points of the next corner
    connected_sides_list = []
    # Each 4 possible corners
    for c_i in range(4):
        connected_side_list = []
        # next corner
        c_j = (c_i + 1) % 4

        candidate_sides = getCandidateSides(corners,c_i,c_j)
        score_list = []
        for side in candidate_sides:
            # score_int = integratePathBtwCorners(side[0],side[1],vx_maps[c_i],vy_maps[c_i])
            score = calculateAffinityBtwCorners(side[0],side[1],vx_maps[c_i],vy_maps[c_i])
            score_list.append(score)
        # Select the best candidate
        while candidate_sides != []:
            # idx_selected = np.argmin(score_list) # Integrate
            
            idx_selected = np.argmax(score_list) # Affinity
            # connected_side_list.append(candidate_sides[idx_selected])
            if score_list[idx_selected] > conf_th:
                # print(score_list[idx_selected])
                side_dict = {'side':candidate_sides[idx_selected], 'score': score_list[idx_selected]}
                connected_side_list.append(side_dict)

            candidate_sides, score_list = deleteCandidates(candidate_sides, score_list, idx_selected)
        connected_sides_list.append(connected_side_list)

        # connected_sides_array = np.array(connected_sides_list)

    return connected_sides_list


    # if np.array_equal(sides[0][0][1],sides[1][0][0]):

    #     print(sides[0][0][1] == sides[1][0][0])

    # else:
    #     print('False')

        # corner_connected_list = []
        # # Each point detected of this corner
        # for p_i in range(len(corners[c_i])):  
        #     # score_corner_list = []
        #     # Next corner
            

def connectGatesFromSides(side_list):

    # side_list = np.array(side_list)
    n_points = []
    for side in side_list:
        n_points.append(len(side))
    n_gates = max(n_points)

    gate_list = []
    for i in range(n_gates):
        gate = {'id':i,'c0':None,'c1':None,'c2':None,'c3':None,'c4':None,'n_corners':0,'score':0}
        # gate = []
        for j in range(4):
            if j == 0:
                if len(side_list[j]) >  i:
                    gate['c0'] = side_list[j][i]['side'][0]
                    gate['c1'] = side_list[j][i]['side'][1]
                else:
                    continue
            else:
                for k in range(len(side_list[j])):
                    c_prev = 'c'+str(j)
                    if np.array_equal(gate[c_prev],side_list[j][k]['side'][0]):
                        c_name = 'c'+str(j+1)
                        gate[c_name] = side_list[j][k]['side'][1]
                        # gate.append(side_list[j][k][1])
                    elif (j == 3) & (np.array_equal(gate['c0'],side_list[j][k]['side'][1])):
                        gate['c3'] = side_list[j][i]['side'][0]
                        gate['c4'] = side_list[j][i]['side'][1]
        # gate = gate[:-1] # Last point is useful just to check if the gate is correct. It may be deleted.
        gate_list.append(gate)

    return gate_list

def checkGates():

    return

def getSides(labels):

    gauss_maps = labels[:4]
    vx_maps = labels[4:8]
    vy_maps = labels[8:]

    corners         = getCornersFromGaussMap(gauss_maps)
    connected_sides = getSidesFromCorners(corners, vx_maps, vy_maps)
    
    return connected_sides

def getGates(connected_sides):

    connected_gates_dict = connectGatesFromSides(connected_sides)

    return connected_gates_dict

    # Alternativa
    # gates = getGatesfromCorners(image_dims, labels)

def detectGates(labels):

    detected_sides = getSides(labels)
    detected_gates = getGates(detected_sides)

    return detected_gates




# def connectSides(corners, vx_maps, vy_maps):

#     # Check points of the next corner
#     connected_sides_list = []
#     # Each 4 possible corners
#     for c_i in range(4):    
#         corner_connected_list = []
#         # Each point detected of this corner
#         for p_i in range(len(corners[c_i])):  
#             # Next corner
#             c_j = (c_i + 1) % 4

#             score_point_list = []
#             # Each point detected of this corner
#             for p_j in range(len(corners[c_j])):  
#                 vx_map = vx_maps[c_i]
#                 vy_map = vy_maps[c_i]
#                 corner_0 = corners[c_i][p_i]
#                 corner_1 = corners[c_j][p_j]
#                 score_point = integratePathBtwCorners(corner_0,corner_1,vx_map,vy_map)
#                 score_point_list.append(score_point)

#             idx_point_selected = np.argmin(score_point_list)
#             corner_connected_list.append([corners[c_i][p_i],corners[c_j][idx_point_selected]])

#         connected_sides_list.append(corner_connected_list)
#         connected_sides_array = np.array(connected_sides_list)

#     return connected_sides_array
