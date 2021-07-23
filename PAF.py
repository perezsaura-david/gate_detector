import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import line
import cv2,math

# def get_corner_from_gauss():
#     # Function to get the points from the gaussian map

#     return

# def generate_PAF():
#     # Function to generate Part Affinity Fields from corner points

#     return

# Make PAF map
def makeVecMaps(img_size, corners, v_points, vector_1, th_dist):

    vx_map = np.zeros((img_size[0],img_size[1]))
    vy_map = np.zeros((img_size[0],img_size[1]))
    
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

def corners2Vector(side_corners):

    # Calculate vector between corner points
    vector = np.array([side_corners[1,0] - side_corners[0,0],side_corners[1,1] - side_corners[0,1]])
    vector_1 = vector / np.linalg.norm(vector)

    # Divide the line in points
    v_points = np.array(list(zip(*line(side_corners[0,0],side_corners[0,1], side_corners[1,0],side_corners[1,1]))))

    return vector_1, v_points

def generatePAF(side_gates, img_size, th_dist = 1):

    vx_map_sum = np.zeros((img_size[0],img_size[1]))
    vy_map_sum = np.zeros((img_size[0],img_size[1]))

    v_points_plot = []
    for side_gate in side_gates:

        vector_1, v_points = corners2Vector(side_gate)

        vx_map, vy_map = makeVecMaps(img_size, side_gate, v_points, vector_1, th_dist)

        vx_map_sum += vx_map
        vy_map_sum += vy_map

        v_points_plot.append(v_points)

    # vx_map_sum = vx_map_sum.transpose()
    # vy_map_sum = vy_map_sum.transpose()

    return vx_map_sum, vy_map_sum, v_points_plot # v(x,y)_map_sum are required. The rest of the variables are for plotting only.


# Plot vec maps
def plotVecMaps(img_size, vx_map, vy_map, side_gates, v_points=[]):
# v0_map = np.zeros((len(xc_grid),len(yc_grid)))
    v0_map = np.zeros_like(vx_map)

    vec_map_f = [vx_map, vy_map]
    vec_map_x = [vx_map, v0_map]
    vec_map_y = [v0_map, vy_map]

    vm = [vec_map_f, vec_map_x, vec_map_y]

    fig, ax = plt.subplots(1,3, figsize=[20,3])

    x = np.arange(0,img_size[0], 1)
    y = np.arange(0,img_size[1], 1)
    xx,yy = np.meshgrid(x,y)

    for i in range(len(ax)):
        ax[i].set_xlim(0,img_size[1])
        ax[i].set_ylim(0,img_size[0])
        ax[i].invert_yaxis()
        
        for points in v_points:
            # Plot subpoints
            ax[i].scatter(points[:,1],points[:,0], c='c', alpha=0.5)
        for corners in side_gates:
            # Plot line
            ax[i].plot(corners[:,1],corners[:,0], c='r')
            # Plot corners points
            ax[i].scatter(corners[:,1],corners[:,0], c='r')
        # Plot vector map
        ax[i].quiver(yy, xx, vm[i][1].transpose(), -vm[i][0].transpose(), scale_units='xy', scale=1, pivot='mid')

    plt.show()

def integratePathBtwCorners():



    return

def points2grid(points, grid_size, c_grid):

    points_grid = [[int(points[0,0]//grid_size),int(points[0,1]//grid_size)]]

    for i in range(len(points)):
        next_point = [int(points[i,0]//grid_size),int(points[i,1]//grid_size)]
        if next_point in points_grid: # Avoid duplicates
            continue
        else:
            points_grid.append(next_point)

    points_grid = np.array(points_grid)

    plot_points = [[points[0,0],points[0,1]]]
    # Plot
    for i in range(len(points)):
        next_point = [int(points[i,0]//grid_size),int(points[i,1]//grid_size)]
        next_center = [c_grid[0][next_point[0]],c_grid[1][next_point[1]]]
        if next_center in plot_points: # Avoid duplicates
            continue
        else:
            plot_points.append(next_center)

    plot_points = np.array(plot_points)

    return points_grid, plot_points

def plotPAFimg(vx_map_sum,vy_map_sum):
    # HSV
    # Hue -> orientation, 
    # Saturation -> const 1
    # Brightness -> magnitude
    hsv_map = np.zeros((vx_map_sum.shape[0],vx_map_sum.shape[1],3), dtype='float32')
    for i in range(vx_map_sum.shape[0]):
        for j in range(vx_map_sum.shape[1]):
            hsv_map[i,j,0] = (math.atan2(vx_map_sum[i,j],vy_map_sum[i,j])+np.pi)*360/(2*np.pi) # Orientation
            hsv_map[i,j,1] = 1.0    # Saturation
            hsv_map[i,j,2] = np.linalg.norm([vx_map_sum[i,j],vy_map_sum[i,j]]) # Magnitude
    
    # Normalize vector magnitude channel
    # hsv_map[:,:,2] = hsv_map[:,:,2] / np.amax(hsv_map[:,:,2])
    # Show image of each channel
    # for i in range(3):
    #     print(np.amax(hsv_map[:,:,i]),np.amin(hsv_map[:,:,i]))

    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2RGB)
    # plt.imshow(vx_map_sum, cmap='hsv')
    plt.imshow(hsv_map)
    plt.show()