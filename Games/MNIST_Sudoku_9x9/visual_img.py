import numpy as np
import matplotlib.pyplot as plt

d1 = np.load("9by9_digit.npy")
d2 = np.load("9by9_upper.npy")

for i in range(10):
    plt.imshow(np.maximum(d1[i][0][0], d2[i][0][0]))
    plt.show()
    #plt.savefig("figs/%d.png"%i)
