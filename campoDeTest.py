from createTags import *
import pdb


PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)


dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='Gaussian')
batch_size = 4
dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# for batch in dl:
#     x,y = batch
#     print(x.shape)

for i,(image,label) in enumerate(dataset):
    label = label.numpy().squeeze()
    # print(label.shape)
    print(i)
    if label.shape[0] != 180:
        print('error')
        pdb.set_trace()
    cv2.imshow('label',label)
    cv2.waitKey()