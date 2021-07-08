import os
from datetime import datetime
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import pytorch_lightning as pl
import torch.nn as nn
from createTags import *

PATH_LABELS  = "./Dataset/training_GT_labels_v2.json"
PATH_IMAGES  = "./Dataset/Data_Training/"
# image_dims = (480,360)
image_dims = (240,180)


class Upsample(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Upsample,self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False,)
        self.bn=nn.BatchNorm2d(channels_out)
        self.relu = nn.ReLU()

        self.deconv = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=(3, 3), stride=(2,2), padding=(1, 1), bias=False,output_padding=(1,1))
    def forward(self, x ):
        x = self.relu(self.bn(self.conv(x)))
        return self.deconv(x)


class GateNet(torch.nn.Module):
    def __init__(self, mode = None):
        super(GateNet,self).__init__()

        if mode is None:
            raise ValueError('mode is not defined.')
        elif mode == 'Vector':
            self.mode = mode 
        elif mode == 'Gaussian' or mode == 'PAFGauss':
            self.mode = 'Gaussian'
        else:
            raise ValueError('mode ',str(mode), ' does not exist.')
        

        # resnet = models.resnet34(pretrained=True)
        resnet = models.resnet18(pretrained=True)
        

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if self.mode == 'Vector':
            self.fc1 = nn.Linear(512*30*15, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 8)

        elif self.mode == 'Gaussian':
            self.up1 = Upsample(512, 512)
            self.up2 = Upsample(512, 256)
            self.up3 = Upsample(256, 256)
            # self.up3 = Upsample(256,1)

            self.conv_1 = nn.Conv2d(256,256,kernel_size = 7,stride = 1,padding = [0,3])
            self.conv_2 = nn.Conv2d(256, 4 ,kernel_size = 3,stride = 1,padding = 1)

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        if self.mode == 'Vector':
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))

        elif self.mode == 'Gaussian':
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = F.relu(self.conv_1(x))
            x = torch.sigmoid(self.conv_2(x))
            

        return x
            
class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss,self).__init__()
    def forward(self, y, y_hat):
        loss = torch.sqrt(torch.mean(torch.pow(y-y_hat,2)))
        return loss


class TrainableGateNet(pl.LightningModule):
    def __init__(self,mode = None):
        super(TrainableGateNet, self).__init__()
        self.mode = mode
        #load dataset
        self.data = gatesDataset(image_dims, PATH_IMAGES, PATH_LABELS,label_transformations='PAFGauss')
        # Divide dataset between train and validation, p is the percentage of data for training
        self.batch_size = 25
           
        p = 0.8
        (self.train_data, self.val_data) = torch.utils.data.random_split(self.data, (
        round(len(self.data) * p), len(self.data) - round(len(self.data) * p)))

        
        # DEFINE MODEL, ALSO DEFINE HEADS HERE
        self.network = GateNet(mode)

        # DEFINE LOSSES
        if mode == 'Vector':
            self.loss_criterion = DetectionLoss()
            print('Detection Loss')
        elif mode == 'Gaussian' or mode == 'PAFGauss':
            # self.loss_criterion = ContinuousFocalLoss()
            self.loss_criterion = nn.MSELoss()
            print('Continuous Loss')
            

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):

        # REQUIRED
        (x, y) = batch
        y_hat = self.forward(x)
        loss = self.loss_criterion(y,y_hat)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        (x, y) = batch
        y_hat = self.forward(x)
        loss = self.loss_criterion(y,y_hat)
        
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     optimizer.step()
    #     optimizer.zero_grad()

    def on_epoch_end(self):
        # if self.current_epoch % 10 == 9:
        RESULTS_PATH = "./checkpoints/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
        PATH = RESULTS_PATH + "_GateNet_" + str(self.current_epoch) + ".pth"
        torch.save(self.state_dict(), PATH)

    # @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=4)



    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)



        
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y


def _neg_loss(gt, pred):
    import cv2
    
    print('gt', gt.shape)
    a = gt.detach().to('cpu')
    b = pred.detach().to('cpu')
    print('pred', pred.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a_img = a[i][j].numpy()*5
            b_img = b[i][j].numpy()*5
            cv2.imshow('gt: '+str(j),a_img)
            cv2.imshow('pred: '+str(j),b_img )
        cv2.waitKey()
        
    '''
    TOOK FROM CENTERNET PAPER 
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
    pred (batch x c x h x w)
    gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.ge(0.8).float()
    neg_inds = gt.lt(0.8).float()
    # pos_inds=gt.gt(0.5).float()
        # print('y_hat shape:',y_hat.
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class ContinuousFocalLoss(nn.Module):
    def __init__(self):
        super(ContinuousFocalLoss, self).__init__()
        self.func=_neg_loss
        # self.beta=beta

    def forward(self, y, y_hat):
        # showMaps(y[0,:,:,:],rescale=True,name='y')
        # showMaps(y_hat[0,:,:,:],rescale=True,name='y_hat')
        # cv2.waitKey()
        # # print('y shape:',y.shape)
        # print('y_hat shape:',y_hat.shape)
        return self.func(y,y_hat)





if __name__ == "__main__":
    net=GateNet('Gaussian')
    x = torch.randn([1,3,image_dims[0],image_dims[1]])
    # print(net(x).shape)


