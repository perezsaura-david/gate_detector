from metrics import getDetectionMetrics
from PAF import detectGates
from PoseEstimation import getCameraParams
from utils import groupCorners, normalizeLabels, resizeLabels, orderCorners, clearLabels
import torch, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from createTags import PAFDataset
from PlotUtils  import showLabels



def showCreatedDataset():


    PATH_LABELS  = "./Dataset/adam_labels_parsed.json"
    PATH_LABELS  = "./Dataset/out.json"
    PATH_IMAGES  = "./Dataset/Data_Adam/"
    # PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    # PATH_IMAGES  = "./Dataset/Data_Training/"
    image_dims = (480,360)

    dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

    for i in tqdm(range(len(dataset))):
        
        # i = 7981
        # i = 8839

        image, labels = dataset[i]
        # image1, labels1 = dataset[i+1]

        # image += image1
        # labels += labels1

        if type(image) == torch.Tensor:
            image = image.numpy() * 255
            image = np.array(image,dtype=np.uint8)
            image = np.transpose(image,(1,2,0))

        if type(labels) == torch.Tensor:
            labels = labels.detach().numpy()

        p = showLabels(image, labels)

        if p == 27 or p == ord('q'):
            break
        else:
            continue

    return

def checkDataset(show):

    PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"

    with open(PATH_LABELS,'r') as json_file:
        labelsDict = json.load(json_file)
    filenames=[]
    for key in labelsDict.keys():
        filenames.append(key)

    fail_list = []
    empty_list = []
    not4_list = []

    for i in tqdm(range(len(filenames))):
        flag_show = False
        image_name = filenames[i]
        labels = labelsDict[image_name]
        points = groupCorners(labels[0])
        points = clearLabels(image_name,points)
        points = orderCorners(points)
        points = np.array(points)
        n_points = len(points)
        if not (n_points > 0):
            flag_show = True
            empty_list.append([image_name])
            continue
        if n_points < 4:
            not4_list.append([image_name], n_points)

        # print(i)
        # for corner in points
        x_mean = np.mean(points[:,:,0])
        y_mean = np.mean(points[:,:,1])
        # x_hlimit = (max(points[:,:,0])+min(points[:,:,0]))/2
        # y_hlimit = (max(points[:,:,1])+min(points[:,:,1]))/2
        fail_count = 0
        for j in range(n_points):
            x = points[j,:,0][0]
            y = points[j,:,1][0]

            if j == 0:
                if not ((x < x_mean) & (y < y_mean)):
                    # print('Bad corner 0')
                    fail_count += 1
            elif j == 1:
                if not ((x > x_mean) & (y < y_mean)):
                    # print('Bad corner 1')
                    fail_count += 1
            elif j == 2:
                if not ((x > x_mean) & (y > y_mean)):
                    # print('Bad corner 2')
                    fail_count += 1
            elif j == 3:
                if not ((x < x_mean) & (y > y_mean)):
                    # print('Bad corner 3')
                    fail_count += 1
        if fail_count > 0:
            fail_list.append([image_name, fail_count])

        if flag_show > 0 or fail_count > 0:

            # SHOW POINTS DISTRIBUTION
            if show == True:
                print(image_name)

                if len(points) > 0:
                    plt.scatter(points[0,:,0],points[0,:,1], c='r')
                if len(points) > 1:
                    plt.scatter(points[1,:,0],points[1,:,1], c='y')
                if len(points) > 2:
                    plt.scatter(points[2,:,0],points[2,:,1], c='g')
                if len(points) > 3:
                    plt.scatter(points[3,:,0],points[3,:,1], c='b')

                plt.axhline(y=y_mean,linewidth=4, color='k')
                plt.axvline(x=x_mean,linewidth=4, color='gray')
                # plt.axhline(y=y_hlimit,linewidth=2, color='r')
                # plt.axvline(x=x_hlimit,linewidth=2, color='b')

            if show == True:
                plt.xlim(0,1200)
                plt.ylim(0,900)
                plt.gca().invert_yaxis()
                plt.show()

    print('fallos',len(fail_list))
    print(fail_list)
    print('vacíos',len(empty_list))
    print(empty_list)
    print('menos de 4 puntos',len(not4_list))
    print(not4_list)




    return


if __name__ == "__main__":

    # Load training dataset
    # PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    # PATH_IMAGES  = "./Dataset/Data_Training/"
    # image_dims = (480,368)

    # with open(PATH_LABELS,'r') as json_file:
    #     labelsDict = json.load(json_file)
    # filenames=[]
    # for key in labelsDict.keys():
    #     filenames.append(key)
    
    # gt = []
    # for i in tqdm(range(len(filenames))):
    #     flag_show = False
    #     image_name = filenames[i]
    #     labels = labelsDict[image_name]
    #     points = groupCorners(labels[0])
    #     gt.append(points)

    # det_metrics = getDetectionMetrics(gt,gt)

    # camera_matrix = getCameraParams()
    # print(camera_matrix)
    # detectGates()

    showCreatedDataset()
    # checkDataset(show=False)
