from metrics import getDetectionMetrics, getImgDetMetrics
import torch, time
from tqdm import tqdm

from GateNet import *
from PlotUtils import addCorners2Image, addDetections2Image, showLabels
from PoseEstimation import getCameraParams

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
device="cpu"

# Load training dataset
PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,368)
test_metrics = True

# dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')


# # Load test dataset
# PATH_IMAGES = "/home/dps/rosbag_gates/images/one_gate/2021-07-09-10-12-46/"
# PATH_IMAGES  = "./Dataset/Data_Test/"
# PATH_IMAGES = "./Dataset/Data_LeaderboardTesting/"
# PATH_IMAGES = "./Dataset/Data_Adam/gates/"
PATH_IMAGES = "/home/dps/rosbag_gates/images/multiple_gates/0/"
# PATH_IMAGES = "/home/dps/rosbag_gates/images/circuit/2021-07-09-11-25-46"
# PATH_LABELS  = "./Dataset/out.json"

test_metrics = False

image_list = os.listdir(PATH_IMAGES)
# image_list.sort()
dataset = image_list

# Load Model
CHECKPOINT_PATH = "checkpoints/September-10-2021_11_36AM_GateNet_10.pth"

net = TrainableGateNet('PAFGauss')
net.load_state_dict(torch.load(CHECKPOINT_PATH))
net = net.to(device)

# camera_matrix, distorsion = getCameraParams()

if test_metrics:
    with open(PATH_LABELS,'r') as json_file:
        labelsDict = json.load(json_file)
    filenames=[]
    for key in labelsDict.keys():
        filenames.append(key)


    image_name = filenames[0]
    image = cv2.imread(os.path.join(PATH_IMAGES,image_name))
    image_h, image_w, image_ch = image.shape
    gt = []
    for i in tqdm(range(len(filenames))):
        flag_show = False
        image_name = filenames[i]
        labels = labelsDict[image_name]
        points = groupCorners(labels[0])
        points = clearLabels(image_name,points)
        points = orderCorners(points)
        points = np.array(points)
        points = normalizeLabels(points, image_w, image_h)
        points = resizeLabels(points, image_dims[0]/2,image_dims[1]/2)
        swap_points = []
        for point in points:
            swap_points.append(point[:,::-1])
        gt.append(swap_points)



avg = 0
detections = []
# for image,label in dataset:
for i in tqdm(range(len(dataset))):
    i = i
    # TEST IMAGES
    # i = 9 # Test gate for PNP
    image_name = dataset[i]
    # image_name = 'onegate_1_frame0376.jpg'
    # image_name = filenames[i]
    image = image2net(image_name, PATH_IMAGES, image_dims)

    # TRAINING IMAGES
    # image, _ = dataset[i]
    # image1, _ = dataset[i+1]
    # image += image1

    net_input = torch.unsqueeze(image,0).to(device)
    
    start_time = time.time()
    output = net(net_input)
    end_time = time.time()
    dif_time = end_time-start_time
    avg = 0.1 * dif_time + 0.9 * avg
    print('time ',dif_time,'avg',avg ,' freq:',1/(avg))

    output=output.detach().to('cpu')[0].numpy()

    if type(image) == torch.Tensor:
        # print(image.shape)
        image = image.numpy() * 255
        image = np.array(image,dtype=np.uint8)
        image = np.transpose(image,(1,2,0))

    p = showLabels(image,output)
    
    if p == 27 or p == ord('q'):
        break
    else:
        continue

    gauss_maps = output[:4]
    corners    = getCornersFromGaussMap(gauss_maps)
    detections.append(corners)

    show_img = True

    if show_img:

        tp_img, fp_img, fn_img, precision_img, recall_img = getImgDetMetrics(corners, gt[i], th_dist=5)

        img_det = factorLabels(corners,2)
        img_gt = factorLabels(gt[i],2)

        image = addDetections2Image(image, img_gt, gt=True)
        image = addDetections2Image(image, img_det, gt=False)
        # image = addImgCenter2Image(image, camera_matrix)
        cv2.imshow('Det vs GT', image)
        p = cv2.waitKey()

        if p == ord('q'):
            break

    # if i == 100:
    #     break



# print('det',np.shape(detections))
# print('gt',np.shape(gt))

# det_metrics = getDetectionMetrics(detections,gt)

