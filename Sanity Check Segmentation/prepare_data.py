import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import numpy 
import matplotlib.pyplot as plt

dir = "data/DIC-C2DH-HeLa/"
data_folders = [ "01","02" ]
label_folders = ["01_ST/SEG/","02_ST/SEG/"]

print("\n\nDATA_FOLDERS")
data = []
for i in data_folders:
    path_i = dir+i
    for j in sorted(os.listdir(path_i)):
        path_j = path_i + "/" + j
        im = Image.open(path_j) 
        imarray = numpy.array(im)
        data.append(imarray)

data= np.array(data)
print(data.shape)
np.save("data.npy",data)

print("\n\nLABEL_FOLDERS")
label = []
for i in label_folders:
    path_i = dir+i
    for j in sorted(os.listdir(path_i)):
        path_j = path_i + "/" + j
        im = Image.open(path_j) 
        imarray = numpy.array(im)
        label.append(imarray)
        

label = np.array(label)
print(label.shape)
np.save("label.npy", label)



