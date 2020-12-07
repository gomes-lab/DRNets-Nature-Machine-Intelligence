import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from skimage.transform import rotate, AffineTransform, warp

def shift(img, offset):
    transform = AffineTransform(translation=offset)
    img = warp(img[0], transform, mode='constant', cval=np.min(img), order=0, preserve_range=True)
    img = img.reshape([1, 32, 32])
    return img 

def diagnose(D, name, path):

    print(D[0][0].shape)
    for i in range(10):
        print(i, len(D[i]), np.mean(D[i]), np.std(D[i]))
        idx = np.arange(len(D[i]))
        np.random.shuffle(idx)

        fig, axes = plt.subplots(10, 10, figsize = (20, 20))
        for k in range(10):
            for l in range(10):
                img = D[i][idx[k*10+l]]
                axes[k][l].imshow(img.reshape((32, 32)))
        plt.savefig(path + "%s_%d"%(name, i))
        plt.clf()
        plt.close()


def select_img(D, n_class, n_sample, offset):
    X = []
    Y = []
    for i in range(n_class):
        for j in range(len(D[i])):
            X.append(D[i][j].reshape(-1))
            Y.append(i)

    X = np.asarray(X)
    Y = np.asarray(Y, dtype = "int32")
    n = len(X)
    print("n_data", n)
    #idx = np.arange(n, dtype="int32")
    #np.random.shuffle(idx)

    #X1 = X[idx[:n//2]]
    #Y1 = Y[idx[:n//2]]

    #X2 = X[idx[n//2:]]
    #Y2 = Y[idx[n//2:]]

    clf = linear_model.LogisticRegression(C=100.0/n, penalty = "l1", solver = "saga", multi_class="multinomial", tol = 0.1)
    clf.fit(X, Y)
    print("CV: acc:", clf.score(X, Y))
    pred = clf.predict_proba(X)
    #clf.fit(X1, Y1)
    #print("CV2: acc:", clf.score(X1, Y1))
    #pred1 = clf.predict_proba(X1)

    #clf.fit(X2, Y2)
    #print("CV1: acc:", clf.score(X2, Y2))
    #pred2 = clf.predict_proba(X2)
    

    #pred = np.concatenate([pred1, pred2], axis = 0)
    #X = np.concatenate([X1, X2], axis = 0)
    #Y = np.concatenate([Y1, Y2], axis = 0)

    print(pred.shape)

    res = {}
    for i in range(n_class):
        res[i] = []

    for i in range(len(X)):
        label = Y[i]
        p = pred[i][label]

        res[label].append([X[i], p])

    rtn = {}
    for i in range(n_class):
        res[i].sort(key = lambda x: -x[1])
        for j in range(10):
            print("%.3f"%res[i][j][1], end = ",")
        print()

        rtn[i] = []
        for j in range(min(n_sample, len(res[i]))):
            img = np.reshape(res[i][j][0], (1, 32, 32))
            img = shift(img, offset)

            #plt.imshow(img[0])
            #plt.show()

            #plt.imshow(img)
            #plt.show()

            rtn[i].append(img)



    return rtn


np.random.seed(19941216)

digits = np.load("digits.npy", allow_pickle = True).item()
uppers = np.load("upper_letters.npy", allow_pickle = True).item()

#diagnose(digits, "digits", "tmp_imgs/")
#diagnose(uppers, "uppers", "tmp_imgs/")

digits = select_img(digits, 10, 4000, (2, 2))

uppers = select_img(uppers, 10, 4000, (-2, -2))

diagnose(digits, "digits", "tmp_imgs2/")
diagnose(uppers, "uppers", "tmp_imgs2/")

np.save("selected_digits_offset2", digits)
np.save("selected_uppers_offset2", uppers)

