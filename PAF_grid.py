import numpy as np
import matplotlib.pyplot as plt


# Make PAF map
def makeVecMaps(grid_size, c_grid, corners_grid, v_points, vector_1, th_dist=2):

    xc_grid = c_grid[0]
    yc_grid = c_grid[1]

    vx_map = np.zeros((len(xc_grid),len(yc_grid)))
    vy_map = np.zeros((len(xc_grid),len(yc_grid)))

    x_corner_grid = corners_grid[:,0]
    y_corner_grid = corners_grid[:,1]

    # Find cells which its center is at some distance from the points
    for i in range(len(vx_map)):
        for j in range(len(vx_map[i])):
            # Limited by the corner points
            if (i < min(x_corner_grid) or i > max(x_corner_grid)):
                continue
            if (j < min(y_corner_grid) or j > max(y_corner_grid)):
                continue
            for x,y in v_points:
                dist = np.sqrt((xc_grid[i]-x)**2+(yc_grid[j]-y)**2)
                grid_dist = dist / grid_size

                if grid_dist < th_dist:
                    if grid_dist < 1:
                        value = 1
                    else:
                        value = 1/th_dist
                    vx_map[i,j] = max(value,vx_map[i,j])
                    vy_map[i,j] = max(value,vy_map[i,j])   

    vx_map = vx_map * vector_1[0]
    vy_map = vy_map * vector_1[1]

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


def corners2Vector(side_corners, dist_subpoints):

    # Calculate vector between corner points
    vector = np.array([side_corners[1,0] - side_corners[0,0],side_corners[1,1] - side_corners[0,1]])
    vector_1 = vector / np.linalg.norm(vector)

    # Divide the line between points in subpoints
    n_points = int(np.linalg.norm(vector) / dist_subpoints)

    v_points = np.zeros((n_points+1,2))
    v_points[0] = side_corners[0,0],side_corners[0,1] # Initial corner
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

    return vector_1, v_points

def generate_PAF(side_gates, img_size, grid_size):

    grid_plot, c_grid = makeGrid(img_size,grid_size) # c_grid is required, grid_plot is just for plot

    vx_map_sum = np.zeros((len(c_grid[0]),len(c_grid[1])))
    vy_map_sum = np.zeros((len(c_grid[0]),len(c_grid[1])))

    v_points_plot = []
    for side_gate in side_gates:

        dist_subpoints = 0.2*grid_size
        # dist_subpoints = 5

        vector_1, v_points = corners2Vector(side_gate, dist_subpoints)

        side_grid = side_gate // grid_size

        vx_map, vy_map = makeVecMaps(grid_size, c_grid, side_grid, v_points, vector_1)

        vx_map_sum += vx_map
        vy_map_sum += vy_map

        # TEST
        v_points, v_plot = points2grid(v_points, grid_size, c_grid)

        v_points_plot.append(v_plot)

    vx_map_sum = vx_map_sum.transpose()
    vy_map_sum = vy_map_sum.transpose()

    return vx_map_sum, vy_map_sum, c_grid, grid_plot, v_points_plot # v(x,y)_map_sum are required. The rest of the variables are for plotting only.


# Plot vec maps
def plotVecMaps(img_size, grid, c_grid, vx_map, vy_map, side_gates, v_points=[]):
# v0_map = np.zeros((len(xc_grid),len(yc_grid)))
    v0_map = np.zeros_like(vx_map)

    vec_map_f = [vx_map,vy_map]
    vec_map_x = [vx_map, v0_map]
    vec_map_y = [v0_map, vy_map]

    vm = [vec_map_f, vec_map_x, vec_map_y]

    fig, ax = plt.subplots(1,3, figsize=[20,4])

    xx,yy = np.meshgrid(c_grid[0],c_grid[1])

    titles = ['vector','componente x', 'componente y']

    for i in range(len(ax)):
        ax[i].set_title(titles[i])
        ax[i].set_xlim(0,img_size[0])
        ax[i].set_ylim(0,img_size[1])
        for x in grid[0]:
            ax[i].axvline(x, linestyle='--', color='gray', linewidth=1, alpha=0.5) # vertical lines
        for y in grid[1]:
            ax[i].axhline(y, linestyle='--', color='gray', linewidth=1 ,alpha=0.5) # horizontal lines
        for points in v_points:
            # Plot subpoints
            ax[i].scatter(points[:,0],points[:,1], c='c', alpha=0.5)
        for corners in side_gates:
            # Plot line
            ax[i].plot(corners[:,0],corners[:,1], c='r')
            # Plot corners points
            ax[i].scatter(corners[:,0],corners[:,1], c='r')
        # Plot vector map
        ax[i].quiver(xx, yy, vm[i][0], vm[i][1], scale_units='xy', scale=0.05, pivot='mid')

    plt.show()

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