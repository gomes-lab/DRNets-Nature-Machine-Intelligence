import numpy as np
import mnist 
from skimage.transform import resize

def rescale(x):
    x = np.asarray(x)
    x = x.reshape(1, 28, 28)
    x = resize(x, (1, 32, 32))
    x = (x - 0.5) / 0.5
    return x

train_images = mnist.train_images()
train_labels = mnist.train_labels()

digit_map = {}
for i in range(10):
    digit_map[i] = []

for i in range(len(train_images)):
    digit_map[train_labels[i]].append(rescale(train_images[i]))

for i in range(10):
    print(i, len(digit_map[i]), digit_map[i][0].shape)

np.save("digits", digit_map)
