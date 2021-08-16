from createTags import PAFDataset
from PlotUtils import showLabels
from tqdm import tqdm

if __name__ == "__main__":

    PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
    PATH_IMAGES  = "./Dataset/Data_Training/"
    image_dims = (480,360)

    dataset = PAFDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')

    for i in tqdm(range(len(dataset))):

        image, labels = dataset[3]

        labels = labels.detach().numpy()

        p = showLabels(image, labels)

        if p == 27 or p == ord('q'):
            break
        else:
            continue