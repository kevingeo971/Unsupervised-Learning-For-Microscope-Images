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

# l = []
for i in tqdm(range(data.shape[0])):
    x = data[i]
    s = seg[i]
    # l+=list(np.unique(s.flatten()))
    for _ in range(50):
        n = np.random.randint(0,512-100)
        crop_data.append( x[n:n+100, n:n+100])
        curr_seg = s[n:n+100, n:n+100]
        print(curr_seg.shape)
        crop_label.append(curr_seg) 


crop_data = np.array(crop_data)
crop_label = np.array(crop_label) 

print(crop_data.shape, crop_label.shape)
X_train, X_val , Y_train, Y_val = train_test_split(crop_data, crop_label, test_size=TestSize, shuffle=True)
print("Shapes : ",X_train.shape, X_val.shape , Y_train.shape, Y_val.shape)

np.save("seg_X_train.npy", X_train )
np.save("seg_X_val.npy", X_val)
np.save("seg_Y_train.npy",Y_train)
np.save("seg_Y_val.npy",Y_val)