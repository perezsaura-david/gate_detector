from createTags import *
import pdb


PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
image_dims = (480,360)


dataset = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')
# batch_size = 4
# dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# for batch in dataset:
#     for obj in batch:
#         print(np.shape(obj))
#     break
# print(x.shape)

# for i,(image,label) in enumerate(dataset):
#     label = label.numpy().squeeze()
#     # print(label.shape)
#     print(i)
#     if label.shape[0] != 180:
#         print('error')
#         pdb.set_trace()
#     cv2.imshow('label',label)
#     k = cv2.waitKey()
#     if k == 27:
#         break

for i, (img, labels) in enumerate(dataset):
    img = np.transpose(img.numpy().squeeze(),[1,2,0])
    cv2.imshow('img', img)
    for j in range(len(labels)):
        lab_name = 'label '+ str(j)
        label = labels[j].numpy().squeeze()
        cv2.imshow(lab_name, label)
    k = cv2.waitKey()
    if k == 27:
        break