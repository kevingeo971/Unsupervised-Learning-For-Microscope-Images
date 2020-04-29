import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

TestSize = 0.1

data = np.load("data.npy")
seg = np.load("label.npy")

print("Shapes : ",data.shape, seg.shape)

crop_data = []
crop_label = []
random_range = (0,512-200) 

l = []
for i in tqdm(range(data.shape[0])):
    x = data[i]
    s = seg[i]
    l+=list(np.unique(s.flatten()))
    
    for _ in range(50):
        n = np.random.randint(0,512-100)
        crop_data.append( x[n:n+100, n:n+100])
        curr_seg = s[n:n+150, n:n+150]
        curr_label = label= np.bincount(curr_seg.flatten()).argmax()
        crop_label.append(curr_label) 


crop_data = np.array(crop_data)
crop_label = np.array(crop_label) 

print(crop_data.shape, crop_label.shape)
print(np.bincount(crop_label), np.unique(crop_label))

uniq = np.unique(crop_label)
dict_lab = { uniq[i]:i for i in range(uniq.shape[0])}
print(dict_lab)

exit()

for i in range(crop_label.shape[0]):
    crop_label[i] = dict_lab[crop_label[i]]

print(np.unique(crop_label))

X_train, X_val , Y_train, Y_val = train_test_split(crop_data, crop_label, test_size=TestSize, shuffle=True)
print("Shapes : ",X_train.shape, X_val.shape , Y_train.shape, Y_val.shape)

np.save( "X_train.npy", X_train )
np.save("X_val.npy", X_val)
np.save("Y_train.npy",Y_train)
np.save("Y_val.npy",Y_val)