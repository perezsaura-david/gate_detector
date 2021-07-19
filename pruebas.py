# import numpy as np
# import matplotlib.pyplot as plt
from utils import *

# Usar coordenadas de una images: origen en esquina superior izquierda
grid_size = 40
img_size = [480,240]

gate1 = [[55,55],[415,155]]
gate2 = [[105,205],[305,155]]

gates_corners = [gate1,gate2]

grid, c_grid = makeGrid(img_size,grid_size)

vx_map_sum = np.zeros((len(c_grid[0]),len(c_grid[1])))
vy_map_sum = np.zeros((len(c_grid[0]),len(c_grid[1])))

# Get corners of every gate in the image
xy_gate_corners  = []
for gate in gates_corners:

    x = []
    y = []
    for point in gate:
        x.append(point[0])
        y.append(point[1])


    xy_gate_corners.append([x,y])

v_points_plot = []
for xy_corners in xy_gate_corners:

    corners_grid, vector_1, v_points = corners2Vector(xy_corners, grid_size)

    vx_map, vy_map = makeVecMaps(grid_size, c_grid, corners_grid, v_points, vector_1)

    vx_map_sum += vx_map
    vy_map_sum += vy_map

    v_points_plot.append(v_points)

vx_map_sum = vx_map_sum.transpose()
vy_map_sum = vy_map_sum.transpose()

plotVecMaps(img_size, grid, c_grid, vx_map_sum, vy_map_sum, xy_gate_corners, v_points_plot)