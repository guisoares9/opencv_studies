import numpy as np

with open('intrinsics.npy', 'rb') as f:
    mtx = np.load(f)

with open('distortion.npy', 'rb') as f:
    dist = np.load(f)

print(mtx)
print(dist)