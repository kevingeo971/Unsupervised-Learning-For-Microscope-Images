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

print("\n\n new Parallel Low LR Dense CAE Gray")

print("\n\nGPU Available : ",torch.cuda.is_available())

print("\n\nNumber of GPU's : ", torch.cuda.device_count())

X_train = np.load("X_train.npy")
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_val = np.load("X_val.npy")
X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)
Y_train = np.load("Y_train.npy")
Y_val = np.load("Y_val.npy")
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

print("\n\nLoaded Data")

class CancerData(Dataset):
    def __init__(self, data, label):
        self.X_data = data
        self.Y_data = label

    def __len__(self):
        return self.X_data.shape[0]

    def transform_image(self, image):
        img_resized_tensor = TF.to_tensor(image)
        normalize_img = (image - 127.5) / 127.5
        return TF.to_tensor(normalize_img).type(torch.FloatTensor)

    def __getitem__(self, idx):
        # print(s\n\nelf.images[idx].shape)
        img_shape = self.X_data.shape
        img_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        return self.transform_image(img_batch), lable_batch

Data_train = CancerData(X_train, Y_train)
Data_val = CancerData(X_val, Y_val)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
Epochs = 200

print("\n - - HyperParameters : ",BATCH_SIZE,LEARNING_RATE,WEIGHT_DECAY,Epochs)

'''
#Best Gray
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
Epochs = 120 

'''

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
X_batch , Y_batch = next(iter(data_loader_val))
print("\n\nOne Batch : ",type(X_batch), X_batch.shape, X_batch.numpy().max(),X_batch.numpy().min(), Y_batch.shape)

class Network(nn.Module):

    def __init__  (self):

        super(Network, self).__init__()

        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(bias=False,in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64)            
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(bias=False,in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 128x32x32
        #self.bn_128 = nn.BatchNorm2d(128)
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
        )

        self.e_conv_4 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        # self.encoded_size = 100352
        # self.lin_in = nn.Linear( self.encoded_size , int(self.encoded_size/32) )
        # self.lin_out = nn.Linear(  int(self.encoded_size/32) , self.encoded_size )

        # Linear

        self.encoded_size = 53824
        self.lin_in = nn.Linear( self.encoded_size , int(self.encoded_size/32) )
        self.lin_out = nn.Linear(  int(self.encoded_size/32) , self.encoded_size )


        #Decoder

        # 128x64x64
        self.d_up_conv_0 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
            
            #nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            
        )

        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),            

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )


        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
            
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),            

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(32)            
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(bias=False,in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),            

            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(bias=False,in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x, encode=False):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1 + ec2
        eblock3 = self.e_block_3(eblock2) + eblock2 + eblock1 + ec2
        eblock4 = self.e_block_3(eblock3) + eblock3 + eblock2 + eblock1 + ec2
        eblock5 = self.e_block_3(eblock4) + eblock4 + eblock3 + eblock2 + eblock1 + ec2
        ec3 = self.e_conv_3(eblock5) 
        ec4 = self.e_conv_4(ec3)
        self.sh = ec4.shape
        self.encoded = self.lin_in(ec4.view(len(ec4), -1))



        if encode:
            return self.encoded

        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded #* 2.0 - 1  # (0|1) -> (-1|1)

        y = self.lin_out( y )
        y = y.view(self.sh)
        uc0 = self.d_up_conv_0(y)
        uc1 = self.d_up_conv_1(uc0)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1 + uc1
        dblock3 = self.d_block_3(dblock2) + dblock2 + dblock1 + uc1
        dblock4 = self.d_block_3(dblock3) + dblock3 + dblock2 + dblock1 + uc1
        dblock5 = self.d_block_3(dblock4) + dblock4 + dblock3 + dblock2 + dblock1 + uc1
        uc2 = self.d_up_conv_2(dblock5)
        dec = self.d_up_conv_3(uc2)

        return dec

model = Network()

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
model.cuda()
# model = nn.parallel.DistributedDataParallel(model)
model = nn.DataParallel(model)

# model = model.cuda()
# min_loss = 10

# x = torch.FloatTensor(X_batch)
# op = model(x.cuda())


best_ep = 0
best_val_acc = -1

def train(epoch):
    
    #Train
    train_acc_sum = 0
    total_loss = 0 
    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        # model.train()
        images, targets = batch[0], batch[1]
        images, targets = images.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,images)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        

    op_train_loss = total_loss / (batch_idx + 1)

    return op_train_loss

def validate(epoch):

    #Validate
    acc_sum = 0
    total_loss = 0 
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_val):
            model.eval()
            images, targets = batch[0], batch[1]
            images, targets = images.cuda(), targets.cuda()
            output = model(images)
            # print(images.cpu().numpy().shape,images.cpu().numpy().shape)
            val_loss = criterion(output,images)
            total_loss+=val_loss.item()

    image = (images[0].cpu().numpy().transpose(1,2,0)*127.5 + 127.5).astype(np.int32)
    output = (output[0].cpu().numpy().transpose(1,2,0)*127.5 + 127.5).astype(np.int32)
    # mim.imsave("Outputs/image.png",image)
    # mim.imsave("Outputs/output.png",output)
    op_val_loss = total_loss / (batch_idx + 1)

    H,W,_ = image.shape
    return op_val_loss, image.reshape((H,W)), output.reshape((H,W))


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss().cuda()

best_ep = 0
best_loss = 100

for i in range(Epochs):
    op_train_loss = train(i)
    op_val_loss,image, output = validate(i)
    print("\n\nEpoch - ",i, " Train Loss : ", op_train_loss," Val Loss : ", op_val_loss, "\n\n")
    if op_val_loss<best_loss:
        best_loss = op_val_loss
        best_ep = i
        torch.save(model.state_dict(), "Best_Cancer_CAE.pwf")
        print("Model saved")
        mim.imsave("Outputs/Best_image.png",image)
        mim.imsave("Outputs/Best_output.png",output)

    else:
        mim.imsave("Outputs/Current_image.png",image)
        mim.imsave("Outputs/Current_output.png",output)

    if i%10==0:
        print("\n\n--------------- Best Values \n\n Epoch - ",best_ep, "Val Accuracy : ", best_loss,"\n\n")

####################
###### RESULTS #####
####################

'''

DDL Network 

'''

