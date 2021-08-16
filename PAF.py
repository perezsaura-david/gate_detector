import torch
import numpy as np


from utils import corners2Vector, getCornersFromGaussMap, normalizeLabels, resizeLabels, integratePathBtwCorners

# def get_corner_from_gauss():
#     # Function to get the points from the gaussian map

#     return

# def generate_PAF():
#     # Function to generate Part Affinity Fields from corner points

#     return

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
                # if dist < th_dist:
                #     vx_map[i,j] = 1
                #     vy_map[i,j] = 1
                # elif dist < 2*th_dist:
                #     vx_map[i,j] = max(0.5,vx_map[i,j])
                #     vy_map[i,j] = max(0.5,vy_map[i,j])
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

            # v_points_plot.append(v_points)

    vx_map_sum = vx_map_sum.transpose()
    vy_map_sum = vy_map_sum.transpose()

    return vx_map_sum, vy_map_sum #, v_points_plot # v(x,y)_map_sum are required. The rest of the variables are for plotting only.

def connectSides(corners, vx_maps, vy_maps):

    # Check points of the next corner
    connected_sides_list = []
    # Each 4 possible corners
    for c_i in range(4):    
        corner_connected_list = []
        # Each point detected of this corner
        for p_i in range(len(corners[c_i])):  
            # score_corner_list = []
            # Next corner
            c_j = (c_i + 1) % 4
            # if c_i == 3:
            #     c_j = 0
            # else:
            #     c_j = c_i + 1

            score_point_list = []
            # Each point detected of this corner
            for p_j in range(len(corners[c_j])):  
                vx_map = vx_maps[c_i]
                vy_map = vy_maps[c_i]
                corner_0 = corners[c_i][p_i]
                corner_1 = corners[c_j][p_j]
                score_point = integratePathBtwCorners(corner_0,corner_1,vx_map,vy_map)
                score_point_list.append(score_point)

            idx_point_selected = np.argmin(score_point_list)
            corner_connected_list.append([corners[c_i][p_i],corners[c_j][idx_point_selected]])

        connected_sides_list.append(corner_connected_list)
        connected_sides_array = np.array(connected_sides_list)

    return connected_sides_array

def connectGatesFromSides(side_list):

    side_list = np.array(side_list)

    gate_list = []
    print(side_list, side_list.shape)
    for i in range(side_list.shape[1]):
        gate = []
        for j in range(4):
            if j == 0:
                gate.append(side_list[j,i,0])
                gate.append(side_list[j,i,1])
            else:
                for k in range(len(side_list[j])):

                    if np.array_equal(gate[-1],side_list[j,k,0]):
                        gate.append(side_list[j,k,1])
        # gate = gate[:-1] # Last point is useful just to check if the gate is correct. It may be deleted.
        gate_list.append(gate)
    
    gates_array = np.array(gate_list)

    return gates_array

def getGates(image_dims, labels):

    gauss_maps = labels[:4]
    vx_maps = labels[4:8]
    vy_maps = labels[8:]

    corners            = getCornersFromGaussMap(gauss_maps)
    normalized_corners = normalizeLabels(corners, image_dims[0],image_dims[1])
    resized_corners    = resizeLabels(normalized_corners, vx_maps.shape[1], vx_maps.shape[2])

    connected_sides = connectSides(resized_corners, vx_maps, vy_maps)

    connected_gates = connectGatesFromSides(connected_sides)

    return connected_gates

    # Alternativa
    gates = getGatesfromCorners(image_dims, labels)

# GRID FUNCTIONS

# # Plot vec maps
# def plotVecMaps(image_dims, vx_map, vy_map, side_gates, v_points=[]):
# # v0_map = np.zeros((len(xc_grid),len(yc_grid)))
#     v0_map = np.zeros_like(vx_map)

#     vec_map_f = [vx_map, vy_map]
#     vec_map_x = [vx_map, v0_map]
#     vec_map_y = [v0_map, vy_map]

#     vm = [vec_map_f, vec_map_x, vec_map_y]

#     fig, ax = plt.subplots(1,3, figsize=[20,3])

#     x = np.arange(0,image_dims[0], 1)
#     y = np.arange(0,image_dims[1], 1)
#     xx,yy = np.meshgrid(x,y)

#     for i in range(len(ax)):
#         ax[i].set_xlim(0,image_dims[1])
#         ax[i].set_ylim(0,image_dims[0])
#         ax[i].invert_yaxis()
        
#         for points in v_points:
#             # Plot subpoints
#             ax[i].scatter(points[:,1],points[:,0], c='c', alpha=0.5)
#         for corners in side_gates:
#             # Plot line
#             ax[i].plot(corners[:,1],corners[:,0], c='r')
#             # Plot corners points
#             ax[i].scatter(corners[:,1],corners[:,0], c='r')
#         # Plot vector map
#         ax[i].quiver(yy, xx, vm[i][1].transpose(), -vm[i][0].transpose(), scale_units='xy', scale=1, pivot='mid')

#     plt.show()



# def points2grid(points, grid_size, c_grid):

#     points_grid = [[int(points[0,0]//grid_size),int(points[0,1]//grid_size)]]

#     for i in range(len(points)):
#         next_point = [int(points[i,0]//grid_size),int(points[i,1]//grid_size)]
#         if next_point in points_grid: # Avoid duplicates
#             continue
#         else:
#             points_grid.append(next_point)

#     points_grid = np.array(points_grid)

#     plot_points = [[points[0,0],points[0,1]]]
#     # Plot
#     for i in range(len(points)):
#         next_point = [int(points[i,0]//grid_size),int(points[i,1]//grid_size)]
#         next_center = [c_grid[0][next_point[0]],c_grid[1][next_point[1]]]
#         if next_center in plot_points: # Avoid duplicates
#             continue
#         else:
#             plot_points.append(next_center)

#     plot_points = np.array(plot_points)

#     return points_grid, plot_points