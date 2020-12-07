import numpy as np

step = 0

def printM(x):
    for i in range(9):
        for j in range(9):
            print(x[i][j]+1, end = " ")
        print()
    print()

def bin_search_acc(label1, res1, label2, res2):
    s_acc = []
    d_acc = []
    for i in range(label1.shape[0]):
        tmp = 0
        for a in range(9):
            for b in range(9):
                p1 = res1[i][a][b]
                l1 = label1[i][a][b]
                r1 = np.argsort(p1)[::-1]
                
                tmp2 = 0
                if (r1[0] == l1 or r1[1] == l1):
                    tmp2 += 1

                p2 = res2[i][a][b]
                l2 = label2[i][a][b]
                r2 = np.argsort(p2)[::-1]
                
                if (r2[0] == l2 or r2[1] == l2):
                    tmp2 += 1
                tmp += tmp2
                d_acc.append(tmp2 == 2) 
        s_acc.append(tmp == 81*2) 
    return np.mean(s_acc), np.mean(d_acc)

def dfs(idx, order, p, D, a): 
    global step
    #print(step)
    
    if (idx == 81):
        step = 999999999999
        return True
    step += 1
    if (step > 100000):
        #print("!!!!!!!!!!!   TLE   !!!!!!!!!!!",step)
        return False

    x, y = order[idx]
    r = np.argsort(p[x][y])[::-1]
    forbiden = []
    
    for (i, j) in order[:idx]:
        if ((i, j) in D[(x,y)]):
            forbiden.append(a[i,j])
            
    #print(x, y, forbiden)
    #printM(a)

    for i in r:
        if (not i in forbiden):
            a[x][y] = i
            t = dfs(idx + 1, order, p, D, a)
            if (t):
                return True
            a[x][y] = -1
    return False
            
def compute_order(p):
    res = []
    for i in range(9):
        for j in range(9):
            res.append((i,j))
    #np.random.shuffle(res)
    #res.sort(key = lambda x: np.max(p[x[0], x[1]]))
    return res

def dfs_search_acc(label1, res1, label2, res2):
    D = {}
    for i in range(9):
        for j in range(9):
            tmp = []
            for k in range(9):
                if (k!=i):
                    tmp.append((k, j))
                if (k!=j):
                    tmp.append((i, k))
            for k in range(3):
                for l in range(3):
                    x = k + (i // 3) * 3
                    y = l + (j // 3) * 3
                    if (x !=i or y !=j):
                        tmp.append((x, y))
            tmp = list(set(tmp))
            D[(i,j)]=tmp
    
    s_acc = [] 
    d_acc = []
    for i in range(label1.shape[0]):
        l1 = np.zeros((9, 9), dtype = "int") - 1
        l2 = np.zeros((9, 9), dtype = "int") - 1
        global step

        step = 0
        #print(step)
        order1 = compute_order(res1[i]) 
        #print(order1)
        t1 = dfs(0, order1, res1[i], D, l1)
        #print(t1, step)
        #printM(l1)
    
        step = 0
        #print(step)
        order2 = compute_order(res1[i]) 
        t2 = dfs(0, order2, res2[i], D, l2)
        #print(t2, step)
        #printM(l2)
        s_acc.append((l1 == label1[i]).all() and (l2 == label2[i]).all())
        #printM(l1)
        #printM(label1[i])
        #printM(l2)
        #printM(label2[i])
        d_acc.append(np.mean(np.logical_and((l1 == label1[i]), (l2 == label2[i]))))

        if (i % 1000 == 0):
            print(i, np.mean(s_acc), np.mean(d_acc))
    return np.mean(s_acc), np.mean(d_acc)
       
"""
path = "resnet-res/"
label1 = np.load(path + "resnet_label1.npy")
label2 = np.load(path + "resnet_label2.npy")
res1 = np.load(path + "resnet_res1.npy")
res2 = np.load(path + "resnet_res2.npy")
"""

"""
path = "DRNet-res/"
label1 = np.load(path + "gt_labels1.npy").reshape([-1, 9, 9]) -1
label2 = np.load(path + "gt_labels2.npy").reshape([-1, 9, 9])
res1 = np.load(path + "pred_probs1.npy").reshape([-1, 9, 9, 9])
res2 = np.load(path + "pred_probs2.npy").reshape([-1, 9, 9, 9])
"""


path = "capsule-res/"
label1 = np.load(path + "capsule_label1_10000.npy")
label2 = np.load(path + "capsule_label2_10000.npy")
res1 = np.load(path + "capsule_res1_10000.npy")
res2 = np.load(path + "capsule_res2_10000.npy")


#print(label2[0])
#print(np.argmax(res2[0], axis = 2))


n = label1.shape[0]
print(label1.shape)
print(res1.shape)
print(label2.shape)
print(res2.shape)

print(set(list(label1.reshape(-1))))
print(set(list(label2.reshape(-1))))

pred1 = np.argmax(res1, axis = 3)
pred2 = np.argmax(res2, axis = 3)

print(np.mean((np.sum(pred1 == label1, axis = (1, 2)) + \
np.sum(pred2 == label2, axis = (1, 2))) == 81 * 2))

print(dfs_search_acc(label1, res1, label2, res2))


