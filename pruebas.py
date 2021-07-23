from PAF import *
from skimage.draw import line
import random, time

# Usar coordenadas de una images: origen en esquina superior izquierda
th_dist = 5
# img_size = [240,480]
img_size = [50,100]
# img_size = [4,8]

# gate1 = [[55,55],[155,415]]
# gate2 = [[205,305],[15,155]]
# gate3 = [[15,25],[35,450]]

gate1 = [[45,5],[15,81]]
gate2 = [[40,60],[20,10]]
gate3 = [[1,2],[40,90]]

gates_corners = [gate1,gate2,gate3]

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

side_gates = np.array(gates_corners)

time0=time.time()
vx_map_sum, vy_map_sum, v_points_plot = generatePAF(side_gates, img_size, th_dist)
time_end = (time.time() - time0)
print('Time spend:', time_end, 's')

plotPAFimg(vx_map_sum,vy_map_sum)

# plotVecMaps(img_size, vx_map_sum, vy_map_sum, side_gates, v_points_plot)
exit()

# Tamaño de celda por pixel. -> Done
# Representar con HSV -> Done
# LineIterator OpenCV Iterar -> Done
# Meter los vertices en un APO

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