import os
from GateNet import *
from pytorch_lightning import Trainer

LOAD_FROM_CHECKPOINT = True

##Train Step
model = TrainableGateNet('PAFGauss')

if LOAD_FROM_CHECKPOINT == True:
    PATH = "./checkpoints/August-04-2021_09_53AM_GateNet_49.pth"
    model.load_state_dict(torch.load(PATH))
    print ('LOADING CHECKPOINT ')

print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
print('and going to http://localhost:6006 on your browser')

trainer = Trainer(default_root_dir='./checkpoints/',max_epochs=100,gpus=1)
trainer.fit(model)