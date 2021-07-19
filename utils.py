import numpy as np
import cv2
import torch
from  createTags import *
import matplotlib.pyplot as plt

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)

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

def groupCorners(points):
    cornerList = []
    
    
    for i in range(len(points)):
        if i%2 == 0:
            x = points[i]
        else:
            y = points[i]
            coord = [x,y]
            cornerPoints = [coord]
            cornerList.append(cornerPoints)
            
            # cornerList.append([[x,y]])
    
    return cornerList

# if __name__ == "__main__":

#     dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)

#     for image,label in dataset:
#         mapa = MakeGaussMap(image,label)
#         cv2.imshow('map',mapa)
#         k = cv2.waitKey()
#         if k == 27:
#             break

def showVecMap():
    # Function to show vectorial maps
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 4)
    y = np.arange(0, 4)

    # vectors x coordinate values
    fx = (0, 1, 1, 2)    
    # vectors y coordinate values
    fy = (1, 0, 1, 2)


    Q = plt.quiver(x, y, fx, fy)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)

    plt.show()
    
    return

def get_corner_from_gauss():
    # Function to get the points from the gaussian map

    return

def generate_PAF():
    # Function to generate Part Affinity Fields from corner points

    return

# Make PAF map
def makeVecMaps(grid_size, c_grid, corners_grid, v_points, vector_1):

    xc_grid = c_grid[0]
    yc_grid = c_grid[1]

    vx_map = np.zeros((len(xc_grid),len(yc_grid)))
    vy_map = np.zeros((len(xc_grid),len(yc_grid)))
    v0_map = np.zeros((len(xc_grid),len(yc_grid)))

    x_corner_grid = corners_grid[0]
    y_corner_grid = corners_grid[1]

    # OPTION 1
    # Find cells that vector crossed
    # for i in range(len(v0_map)):
    #     for j in range(len(v0_map[i])):
    #         if [i,j] in v_points_grid:
    #             vx_map[i,j] = 1
    #             vy_map[i,j] = 1
    # OPTION 2
    # Find cells which its center is at some distance from the points
    for i in range(len(v0_map)):
        for j in range(len(v0_map[i])):
            # Limited by the corner points
            if (i < min(x_corner_grid) or i > max(x_corner_grid)):
                continue
            if (j < min(y_corner_grid) or j > max(y_corner_grid)):
                continue
            for x,y in v_points:
                dist = np.sqrt((xc_grid[i]-x)**2+(yc_grid[j]-y)**2)
                if dist < grid_size:
                    vx_map[i,j] = 1
                    vy_map[i,j] = 1
                elif dist < 2*grid_size:
                    vx_map[i,j] = max(0.5,vx_map[i,j])
                    vy_map[i,j] = max(0.5,vy_map[i,j])

    vx_map = vx_map * vector_1[0]
    vy_map = vy_map * vector_1[1]

    # vx_map = vx_map.transpose()
    # vy_map = vy_map.transpose()

    return vx_map, vy_map

def makeGrid(img_size,grid_size):
    # Make grid
    x_grid = np.arange(0,img_size[0]+grid_size, grid_size)
    y_grid = np.arange(0,img_size[1]+grid_size, grid_size)

    grid = [x_grid, y_grid]
    # Find centers of the grid cells
    xc_grid = np.zeros(len(x_grid)-1)
    yc_grid = np.zeros(len(y_grid)-1)

    for i in range(len(xc_grid)):
        xc_grid[i] = (x_grid[i] + x_grid[i+1])/2
    for i in range(len(yc_grid)):
        yc_grid[i] = (y_grid[i] + y_grid[i+1])/2

    c_grid = [xc_grid,yc_grid]

    return grid, c_grid

def corners2Vector(xy_corners, grid_size):
    x_corners = xy_corners[0]
    y_corners = xy_corners[1]

    # Convert corner coords in grid coords
    x_corner_grid = []
    y_corner_grid = []
    for x in xy_corners[0]:
        x_corner_grid.append(x//grid_size)
    for y in xy_corners[1]:
        y_corner_grid.append(y//grid_size)

    corners_grid = [x_corner_grid,y_corner_grid]

    # Calculate vector between corner points
    vector = [x_corners[1] - x_corners[0], y_corners[1] - y_corners[0]]
    vector_1 = vector / np.linalg.norm(vector)

    # Find cells crossed by the vector
    dist_subpoints = 0.2*grid_size
    # Divide the line between points in subpoints
    n_points = int(np.linalg.norm(vector) / dist_subpoints)

    v_points = np.zeros((n_points+1,2))
    v_points[0] = x_corners[0],y_corners[0]
    for i in range(n_points):
        vx_next_point = v_points[i,0] + vector_1[0]*dist_subpoints
        vy_next_point = v_points[i,1] + vector_1[1]*dist_subpoints
        v_points[i+1] = vx_next_point, vy_next_point

    # # Translate points to grid coords
    # v_points_grid = [[x_corner_grid[0],y_corner_grid[0]]]

    # for i in range(len(v_points)):
    #     next_point = [v_points[i,0]//grid_size,v_points[i,1]//grid_size]
    #     if next_point != v_points_grid[-1]: # Avoid duplicates
    #         v_points_grid.append(next_point)

    return corners_grid, vector_1, v_points 

# Plot vec maps
def plotVecMaps(img_size, grid, c_grid, vx_map, vy_map, corners, v_points=[]):
# v0_map = np.zeros((len(xc_grid),len(yc_grid)))
    v0_map = np.zeros_like(vx_map)

    vec_map_f = [vx_map,vy_map]
    vec_map_x = [vx_map, v0_map]
    vec_map_y = [v0_map, vy_map]

    vm = [vec_map_f, vec_map_x, vec_map_y]
    
    x_corners = corners[0]
    y_corners = corners[1]


    fig, ax = plt.subplots(1,3, figsize=[20,4])

    xx,yy = np.meshgrid(c_grid[0],c_grid[1])

    for i in range(len(ax)):
        ax[i].set_xlim(0,img_size[0])
        ax[i].set_ylim(0,img_size[1])
        for x in grid[0]:
            ax[i].axvline(x, linestyle='--', color='gray', linewidth=1, alpha=0.5) # vertical lines
        for y in grid[1]:
            ax[i].axhline(y, linestyle='--', color='gray', linewidth=1 ,alpha=0.5) # horizontal lines
        for points in v_points:
            # Plot subpoints
            ax[i].scatter(points[:,0],points[:,1], c='c', alpha=0.5)
        for x_corners,y_corners in corners:
            # Plot line
            ax[i].plot(x_corners,y_corners, c='r')
            # Plot corners points
            ax[i].scatter(x_corners,y_corners, c='r')
        # Plot vector map
        ax[i].quiver(xx, yy, vm[i][0], vm[i][1], scale_units='xy', scale=0.05, pivot='mid')

    plt.show()