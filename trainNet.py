import  os
from GateNet import *
from pytorch_lightning import Trainer

LOAD_FROM_CHECKPOINT = False

##Train Step

model = TrainableGateNet('Gaussian')

# if LOAD_FROM_CHECKPOINT == True:
#     PATH = "./checkpoints/November-05-2019_16_36PM_test_batch_centers4.pth"
#     model.load_state_dict(torch.load(PATH))

print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
print('and going to http://localhost:6006 on your browser')

# trainer = Trainer(default_root_dir='./checkpoints/',max_nb_epochs=1000,gpus=1)
trainer = Trainer(default_root_dir='./checkpoints/',max_epochs=1000,gpus=1)
trainer.fit(model)
