from collections import defaultdict
import mnist
import random
import numpy as np
import sys
from skimage.transform import resize

def rescale(x):
    x = np.asarray(x)
    x = x.reshape(16, 1, 28, 28)
    x = np.transpose(x, (1, 2, 3, 0))
    x = resize(x, (1, 32, 32, 16))
    x = np.transpose(x, (3, 0, 1, 2))
    for i in range(16):
        x[i][0] = (x[i][0] - 0.5) / 0.5
        
    return x

def main():
    seed = int(sys.argv[4])
    np.random.seed(seed)
    # the number of testing examples
    n = int(sys.argv[1])
    size = int(sys.argv[2])
    if ("5" in sys.argv[3]):
        diff_num = True
    else:
        diff_num = False

    if size == 4:
        sudokus = np.load("all4sudoku.npy")
    elif size == 9:
        sudokus = np.load("minimum.npy")[:, 1, :, :]

    if diff_num and size == 4:
        digit2digit = {1: 5, 2: 6, 3: 7, 4: 8}
        for idx, num in np.ndenumerate(sudokus):
            sudokus[idx] = digit2digit[num]

    test_images = mnist.test_images() #mnist.test_images()
    #test_images = [cv2.resize(img, (32, 32)) for img in test_images]
    test_labels = mnist.test_labels() #mnist.test_labels()

    digit_map = defaultdict(list)

    for i in range(len(test_labels)):
        digit_map[test_labels[i]].append(test_images[i])

    rtn = []
    rtn_labels = []
    n_iter = n//len(sudokus)+1
    for i in range(n_iter):  # 35*288 is roughly 10000
        for sudoku in sudokus:
            flatten = [number for sublist in sudoku for number in sublist]
            rtn_labels.append(flatten)
            mnist_sudoku = []
            for number in flatten:
                rnd = np.random.randint(len(digit_map[number]))
                mnist_sudoku.append(digit_map[number][rnd])

            rtn.append(rescale(mnist_sudoku))

    rtn, rtn_labels = rtn[:n], rtn_labels[:n]
    rtn, rtn_labels = np.array(rtn), np.array(rtn_labels)


    print(rtn.shape)
    print(rtn_labels.shape)
    suffix = ""
    if (diff_num):
        suffix = "_5678"
    
    s_rtn = []
    s_rtn_labels = []
    idx = np.arange(len(rtn))
    np.random.shuffle(idx)
    for i in idx:
        s_rtn.append(rtn[i])
        s_rtn_labels.append(rtn_labels[i])

    np.save("{0}by{0}{1}.npy".format(size, suffix), s_rtn)
    np.save("{0}by{0}{1}_labels.npy".format(size, suffix), s_rtn_labels)

if __name__ == '__main__':
    main()
