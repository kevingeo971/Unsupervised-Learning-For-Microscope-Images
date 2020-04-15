print("\n\nTrain CAE Unet")

import os
from tqdm import tqdm
import cv2 
import numpy as np
import matplotlib.image as mim
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from sklearn.model_selection import train_test_split
from unet import UNet

print("\n\n  W Net Weighted")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X_train_lowF = np.load("X_train_lowF.npy")
X_val_lowF = np.load("X_val_lowF.npy")

X_train_og = np.load("X_train_og.npy")
X_val_og = np.load("X_val_og.npy")

X_train_highF = np.load("X_train_highF.npy")
X_val_highF = np.load("X_val_highF.npy")

print("Shape : ",X_train_og.shape, X_train_lowF.shape, X_train_highF.shape)

print("\n\nLoaded Data")

class CancerData(Dataset):
    def __init__(self, data, lowF, highF):
        self.X_data = data
        self.X_data_lowF = lowF
        self.X_data_highF = highF

    def __len__(self):
        return self.X_data.shape[0]

    def transform_image(self, image):
        img_resized_tensor = TF.to_tensor(image)
        normalize_img = (image - 127.5) / 127.5
        return TF.to_tensor(normalize_img).type(torch.FloatTensor)

    def __getitem__(self, idx):
        og_batch = self.X_data[idx]
        lowF_batch = self.X_data_lowF[idx]
        highF_batch = self.X_data_highF[idx]
        return self.transform_image(og_batch), self.transform_image(lowF_batch), self.transform_image(highF_batch)

Data_train = CancerData(X_train_og, X_train_lowF, X_train_highF)
Data_val = CancerData(X_val_og, X_val_lowF, X_val_highF)

BATCH_SIZE = 32
LEARNING_RATE = 1e-2
LR_DECAY = 0.80
WEIGHT_DECAY = 0.000
Epochs = 1000
WEIGHT_HIGH = 0.6

print("\n - - HyperParameters : ",BATCH_SIZE,LEARNING_RATE,LR_DECAY,WEIGHT_DECAY,Epochs)

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
X_batch , X_batch_lowF, _ = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min())

model = UNet(in_channels=1, out_channels=1, only_encode=False)
     
#def weights_init(m):
#    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
#        nn.init.xavier_uniform_(m.weight.data)
        #nn.init.xavier_uniform_(m.bias.data)

#model.apply(weights_init)
'''
#
#Data Parallel ---------------------------------------------
def get_ifname():
    return ifcfg.default_interface()["device"]

if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

if "NCCL_SOCKET_IFNAME" not in os.environ:
    os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

master_port = int(os.environ.get("MASTER_PORT", 8738))
master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

tcp_store = torch.distributed.TCPStore(master_addr, master_port, world_size, world_rank == 0)
torch.distributed.init_process_group(
    "nccl", store=tcp_store, rank=world_rank, world_size=world_size
)
#-----------------------------------------------------------
'''

# model = nn.parallel.DistributedDataParallel(model)

model = nn.DataParallel(model).cuda()

best_ep = 0
best_val_acc = -1

def train(epoch):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        model.train()
        images, low , high = batch[0], batch[1], batch[2]
        images, low , high = images.cuda(), low.cuda(), high.cuda()
        
        optimizer.zero_grad()
        output_low, output_high = model(images)
        loss_low = criterion(output_low, low)
        loss_high = criterion(output_high, high)
        loss = WEIGHT_HIGH*loss_low + (1-WEIGHT_HIGH)*loss_high
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        
    op_train_loss = total_loss / (batch_idx + 1)

    return op_train_loss

def validate(epoch):

    #Validate
    acc_sum = 0
    total_loss = 0 
    for batch_idx, batch in enumerate(data_loader_val):
        model.eval()
        images, low , high = batch[0], batch[1], batch[2]
        images, low , high = images.cuda(), low.cuda(), high.cuda()
        
        optimizer.zero_grad()
        with torch.no_grad():
            output_low, output_high = model(images)
        loss_low = criterion(output_low, low)
        loss_high = criterion(output_high, high)
        val_loss = WEIGHT_HIGH*loss_low + (1-WEIGHT_HIGH)*loss_high
        total_loss+=val_loss.item()

    image = (images[0].cpu().numpy()*127.5 + 127.5).astype(np.int32)
    output_low = (output_low[0].cpu().numpy()*127.5 + 127.5).astype(np.int32)
    output_high = (output_high[0].cpu().numpy()*127.5 + 127.5).astype(np.int32)
    gt_low = (low[0].cpu().numpy()*127.5 + 127.5).astype(np.int32)
    gt_high = (high[0].cpu().numpy()*127.5 + 127.5).astype(np.int32)
    
    
    # mim.imsave("Outputs/image.png",image)
    # mim.imsave("Outputs/output.png",output)
    op_val_loss = total_loss / (batch_idx + 1)

    return op_val_loss, image, output_low, output_high, gt_low, gt_high


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = nn.MSELoss().cuda()

best_ep = 0
best_loss = 100

for i in range(Epochs):
    op_train_loss = train(i)
    op_val_loss,image, output_low, output_high, gt_low, gt_high = validate(i)
    scheduler.step()
    print("\n\nEpoch - ",i, " Train Loss : ", op_train_loss," Val Loss : ", op_val_loss, "\n\n")
    if op_val_loss<best_loss:
        best_loss = op_val_loss
        best_ep = i
        torch.save(model.state_dict(), "Best_Cancer_CAE.pwf")
        print("Model saved")
        mim.imsave("Outputs/Best_image.png",image.squeeze())
        mim.imsave("Outputs/Best_output_low.png",output_low.squeeze())
        mim.imsave("Outputs/Best_output_high.png",output_high.squeeze())
        mim.imsave("Outputs/Best_gt_low.png",gt_low.squeeze())
        mim.imsave("Outputs/Best_gt_high.png",gt_high.squeeze())

    else:
        mim.imsave("Outputs/Current_image.png",image.squeeze())
        mim.imsave("Outputs/Current_low.png",output_low.squeeze())
        mim.imsave("Outputs/Current_high.png",output_high.squeeze())

    if i%10==0:
        print("\n\n--------------- Best Values \n\n Epoch - ",best_ep, "Val Accuracy : ", best_loss,"\n\n")

