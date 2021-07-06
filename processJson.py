# Borrable
import cv2,torch
import numpy as np
from createTags import gatesDataset,plotGates
import  pdb

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,240)


if __name__ == "__main__":

    data = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS)
    #
    # for i,(a,b) in enumerate(data):
    #     if i%100 == 0 :
    #         print(i,a.shape,b.shape)
    #
    #     if (a.shape[0] != 3) or (a.shape[1] < 100) or (a.shape[2] < 100):
    #         print('image_failed')
    #         pdb.set_trace()
    #
    #     if (b.shape[0] != 1) or (b.shape[1] != 8) :
    #         print('label_failed')
    #         pdb.set_trace()

    # c,d = data[1057]
    dl_torch = torch.utils.data.DataLoader(gatesDataset(image_dims,PATH_IMAGES,PATH_LABELS), batch_size=5, shuffle=True, num_workers=0)

    for i,batch in enumerate(dl_torch):
        batch_images, batch_labels  = batch
        # print(i)
        # for i in range(len(batch_images)):
            # plotGates(batch_images[i],batch_labels[i])
            # print(batch_labels)

    