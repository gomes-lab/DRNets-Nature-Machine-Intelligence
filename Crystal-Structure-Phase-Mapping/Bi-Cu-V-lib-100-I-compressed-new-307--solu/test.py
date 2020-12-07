import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import model
import get_data 
import config 
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import urllib
import os, sys
#from pyheatmap.heatmap import HeatMap
#import seaborn as sns

FLAGS = tf.app.flags.FLAGS
def stats(name, x, visual = True):
    idx = np.argsort(x)
    print("stats of %s min:"%name, np.min(x), "max:", np.max(x), "mean", np.mean(x), "median", np.median(x), "std", np.std(x), "worse_idx", idx[-1])
    if (visual):
        plt.hist(x, 100)
        plt.title(name)
        plt.show()
    return idx

def rescale(x):
    return x / (FLAGS.eps + np.max(x))
    #* FLAGS.peak_rescale  #/ (1e-9 + np.max(x))

def compute_XRD(mu, intensity):
    Q = np.load(FLAGS.Q_dir)
    y = np.zeros_like(Q)
    x = Q 
    max_I = np.max(intensity)
    for i in range(FLAGS.n_spike):
        pos = int((mu[i] - Q[0])/(Q[-1] - Q[0])*Q.shape[0])
        if (pos >= 0 and pos < FLAGS.xrd_dim):
            y[pos] += intensity[i]/max_I

    return y

def visual_bases_sol(bases_sol):
    M = bases_sol.shape[0]
    #print("M=", M)
    f, axes = plt.subplots(M, 1, figsize = (15, M*2.5))
    for i in range(M):
        ax = axes[i]
        x = bases_sol[i]
        x = rescale(x)
        ax.plot(x)
    plt.show()
    #plt.savefig("bases.png")

def comp2Coords(comp):

    vec2 = np.array([0,0])
    vec3 = np.array([1,0])
    vec1 = np.array([1.0/2, np.sqrt(3.0/4)])
    array = []


    for i in range(len(comp)):
        array.append(comp[i][0]*vec1 + comp[i][1]*vec2 + comp[i][2]*vec3)

    return np.asarray(array) # (x_i, y_i)
    #x = [item[0] for item in array]
    #y = [item[1] for item in array]

    #return x,y

def sticksPattern(basis, mu_shift, Q, rescale = True):

    mu_init = np.zeros(200)
    c = np.zeros(200)

    for (i, peak) in enumerate(basis):
        mu_init[i] = peak[0] #(peak[0] - ((80 + 15) / 2.0))/(80 - 15) * 2.0 * lim
        c[i] = peak[1]

    c /= np.max(c)
    mu = mu_init * mu_shift
    eps = 1e-9

    return [mu, c]

def plot_recon(xrd, JS_dis_batch, L2_dis_batch, degree_of_freedom, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch, alloy_loss_batch, basis_weights, avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity):

    top_bases = np.load("exist_bases.npy")
    shift_status = np.load("shift_status.npy")
    max_shifts = np.load("max_shifts.npy")
    Q_idx = np.load(FLAGS.Q_idx_dir)
    Q = np.load(FLAGS.Q_dir)
    bases_name = np.load(FLAGS.bases_name_dir)
    stick_bases = np.load(FLAGS.stick_bases_dir)

    visual = True
    if ("single" in sys.argv):
        visual = False

    idx1 = stats("recon_loss", recon_loss, visual)[::-1]
    stats("gibbs_loss", gibbs_loss_batch, visual)
    
    print(basis_weights.shape)

    top_bases = np.load("exist_bases.npy")
    top_bases = np.sort(top_bases)
    #idx2 = stats("weights_L1_loss", np.sum(np.abs(basis_weights[:, :M] - weights_sol), axis = 1))
    found_bases = [80, 67, 47, 55, 46, 60, 40, 35, 5, 76, 1, 63, 62, 12, 22]
    #print(idx1)
    best_examples = list(idx1[-2:])
    worst_examples = list(idx1[:100])


    examples = best_examples + worst_examples

    if ("binary" in sys.argv):
        num = int(sys.argv[-1])
        binary_examples = []
        for i in range(degree_of_freedom.shape[0]):
            if (composition[i][num] == 0):
                binary_examples.append(i)
        binary_examples.sort(key = lambda x: tuple(list(composition[x])))
        examples = binary_examples

    if ("single" in sys.argv):
        examples = [int(sys.argv[-1]) - 1]

    if ("basis" in sys.argv):
        examples = np.argsort(basis_weights[:,int(sys.argv[-1])])[::-1]

    if ("savefig" in sys.argv):
        if (not "binary" in sys.argv):
            examples = np.arange(basis_weights.shape[0])
        dir = 'figs'
        if not os.path.exists(dir):
            os.makedirs(dir)
    snt = 0
    for i in examples:
        snt += 1
        #plt.clf()
        n_fig =  3 + 3
        f, axes = plt.subplots(n_fig, 2, figsize = (10*2, n_fig*2.5))
        #print(axes)
        #dcp = (decomposition_sol[i].T / (1e-9 + np.max(decomposition_sol[i], axis = 1)) )
        #xrd_recon = np.sum(decomposition_sol[i], axis = 0) #np.dot(dcp, weights_sol[i])
        #xrd_std = xrd[i] #/ np.max(xrd[i])
        #print(np.sum(np.abs(xrd_std - xrd_recon)))

        Xs1 = [xrd[i][Q_idx]] #, ]
        legends1 = ["xrd_std of #%d"%(i + 1)] #, 
        title1 =  "xrd_std of #%d, composition = %s"%(i + 1, composition[i]) #, 
        axes[0][0].set_title(title1)
        axes[0][0].plot(Q, rescale(Xs1[0]), color = "red")
        axes[0][0].legend([legends1[0]], loc = "upper right")


        Xs2 = [xrd_prime[i]]
        legends2 = ["xrd_recon: %.6f"%(recon_loss[i])]
        
        title2 ="comp_loss = %.3f [%.2f, %.2f, %.2f]"%(comp_loss_batch[i], comp_prime[i][0], comp_prime[i][1], comp_prime[i][2])
        axes[0][1].set_title(title2)
        #sticks = [0]
        idx = np.argsort(basis_weights[i])[::-1]
        idx = idx[:3]

        stick_patterns = []
        cnt = 0
        for j in idx[:3]:
            flag = 1
            if ("refine" in sys.argv):
                flag = (j in top_bases)

            if (basis_weights[i][j] >= 1e-3 and flag):
                Xs2.append(decomp[i][j])

                stick_patterns.append(sticksPattern(stick_bases[j], mu_shift[i][j], Q))
                legends2.append("basis-%d, weight =%.3f, mu_shift=%.4f, logvar=%.3f"%(j, basis_weights[i][j], mu_shift[i][j], logvar[i][j]))
                cnt += 1
                axes[cnt][0].plot([],[], ' ')
                axes[cnt][0].legend([bases_name[j][:-4]])



            #legends2.append("basis-%d, weight =%.3f, logvar=%.3f"%(j, basis_weights[i][j], logvar[i][j][0]))
            #legends2.append("basis-%d, weight =%.6f"%(j, basis_weights[i][j]))
            #sticks.append(compute_XRD(mu[i][j], intensity[i][j]))

        axes[0][1].plot(Q, rescale(Xs1[0]), color="red")
        axes[0][1].plot(Q, rescale(Xs2[0]))


        info = ['shift_status = %d'%(shift_status[i]), 'max_shift = %.6f'%(max_shifts[i]), 'gibbs_loss = %.2f'%gibbs_loss_batch[i], "degree_of_freedom: %.1f"%degree_of_freedom[i]]
        info += ["JS_dis = %.2f"%JS_dis_batch[i], "L2_dis = %.2f"%L2_dis_batch[i]]
        for z in info:
            axes[-2][1].plot([], [], ' ')
        axes[-2][1].legend(info)
        
        tmp = ["xrd_std", legends2[0]]
        axes[0][1].legend(tmp, loc = "upper right")
        



        axes[-1][0].plot(Q, rescale(Xs2[0]))
        axes[-1][0].legend(["VAE recon phase"], loc = "upper right")

        cnt = 0
        for j in range(1, len(Xs2)):
            if ("stick" in sys.argv):
                mu, c = stick_patterns[cnt]
                for k in range(len(mu)):
                    if (c[k] > 1e-3):
                        axes[j][1].plot([mu[k], mu[k]],  [0.0, c[k]], c = "orange")
            #axes[j][1].plot(Q, x, c = "orange")


            axes[j][1].plot(Q, rescale(Xs2[j]))
            axes[j][1].legend([legends2[j]])
            cnt += 1
            #axes[j][1].plot(sticks[j])
            #axes[j][1].set_ylim(0, 10)

        axes[-1][1].plot(Q, noise)
        axes[-1][1].set_ylim(0, 1)
        axes[-1][1].legend(["noise mean = %.6f"%(np.mean(noise))], loc = "upper right")
        if ("savefig" in sys.argv):
            s = "sample"
            if ("binary" in sys.argv):
                s = str(snt)
            plt.savefig("figs/%s_%d.png"%(s, i + 1))
            plt.close('all')
            print("saving fig-%d"%(i + 1))
        else:
            plt.show()

def plot_ternary(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch, avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity):
    
    mean_basis_weights = np.mean(basis_weights, axis = 0)
    top_bases = np.load("exist_bases.npy")
    top_bases = np.sort(top_bases)
    n_top_bases = len(top_bases)


    N = xrd.shape[0]
    M = basis_weights.shape[1]
    print("N = %d, M = %d"%(N, M))
    colors = [] # 8
    n_interval = 1
    while (np.power(n_interval, 3) - 1 < n_top_bases):
        n_interval += 1

    print("n_interval = ", n_interval)

    for i in range(n_interval):
        for j in range(n_interval):
            for k in range(n_interval):
                tmp = np.asarray([i, j, k]) * 1.0 / (n_interval - 1)
                colors.append(tmp)

    colors = np.asarray(colors, dtype= "float32")[:-1] # throw out the "white"
    np.random.shuffle(colors)
    colors = colors[:n_top_bases]


    color_map = {}
    for (i, b) in enumerate(top_bases):
        color_map[b] = i


    n_col = 3 
    n_row = int((n_top_bases -1) / n_col)  +1
    fig, axes = plt.subplots( n_row, n_col, figsize = (3.5*n_col, 3.5*n_row))
    annot = np.zeros_like(axes).reshape([-1])
    sc = np.zeros_like(axes).reshape([-1])

    for j in range(n_top_bases):
        coords = comp2Coords(composition)
        X = coords[:, 0]
        Y = coords[:, 1]

        u = int(j / n_col)
        v = int(j % n_col)

        b = top_bases[j]

        sc[j] = axes[u][v].scatter(X, Y, s = (6.0)**2 , c = "#eeefff")
        annot[j] = axes[u][v].annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot[j].set_visible(False)
        axes[u][v].set_gid(j)
        
        rgb = np.tile([colors[j]], [len(X), 1])

        axes[u][v].scatter(X, Y, s = basis_weights[:, b]*(6.0)**2 , c = rgb) #
        axes[u][v].legend(["%d, %.3f"%(b, mean_basis_weights[b])], loc = "upper right")
        axes[u][v].set_title("recon map")
        

    def update_annot(ind, gid):
        pos = sc[gid].get_offsets()[ind["ind"][0]]
        annot[gid].xy = pos
        text = str(ind["ind"][0] + 1)
        annot[gid].set_text(text)
        annot[gid].get_bbox_patch().set_facecolor("black")
        annot[gid].get_bbox_patch().set_alpha(0.4)

    def hover(event):
        if (event.inaxes!= None):
            gid = event.inaxes.get_gid()
            if (gid != None and gid < n_top_bases):
                vis = annot[gid].get_visible()
                cont, ind = sc[gid].contains(event)
                if cont:
                    update_annot(ind, gid)
                    annot[gid].set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot[gid].set_visible(False)
                        fig.canvas.draw_idle()
        else:
            for gid in range(n_top_bases):
                annot[gid].set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


    plt.show()



    edges = np.load(FLAGS.edges_dir)

    fig, axes = plt.subplots(1, 1, figsize = (12, 10))
    coords = comp2Coords(composition)
    X = coords[:, 0]
    Y = coords[:, 1]
    axes.scatter(X, Y, s = (15.0)**2 , c = "#eeefff", alpha = 0.5)

    sample_colors = []
    legend_elements = []

    phase_dict = {}
    phase_arr = []
    for i in range(N):
        w = basis_weights[i]
        c = []
        phase = []

        for j in range(M):
            if (w[j] >= FLAGS.active_th and j in top_bases):
                c.append(colors[color_map[j]])
                phase.append(j)

        c = np.mean(c, axis = 0)
        sample_colors.append(c)
        phase = str(phase)
        phase_arr.append(phase)
        if (not phase in phase_dict):
            phase_dict[phase] = 1
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label='phase %s'%phase, markerfacecolor=c, markersize=10))
    
    for i in range(N):
        for j in range(N):
            if (edges[i][j] == 1):
                if (phase_arr[i] == phase_arr[j]):
                    #print(i, j)
                    axes.plot([X[i], X[j]],  [Y[i], Y[j]], c = "black", alpha= 0.5, zorder = -1)

                #axes.plot([X[i], X[j]],  [Y[i], Y[j]], c = "black", linestyle='dashed')

    sc = axes.scatter(X, Y, s = (10.0)**2 , c = sample_colors , zorder = 1) #
    annot = axes.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot2(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = str(ind["ind"][0] + 1) + "-" + phase_arr[int(ind["ind"][0])]
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor("white")
        annot.get_bbox_patch().set_alpha(0.9)

    def hover2(event):
        vis = annot.get_visible()
        if event.inaxes == axes:
            cont, ind = sc.contains(event)
            if cont:
                update_annot2(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                     
    #axes.legend(handles = legend_elements, loc = 'upper right')
    axes.set_title("phase mapping")
    fig.canvas.mpl_connect("motion_notify_event", hover2)
    plt.show()

def is_connected(edges, v):
    s = -1 
    for i in range(v.shape[0]):
        if (v[i] == 1):
            s = i 
            break
    vis = np.zeros_like(v)
    vis[s] = 1
    st = []
    st.append(s)
    head = 0
    while (head < len(st)):
        v_now = st[head]
        head += 1
        for i in range(edges.shape[1]):
            if (edges[v_now][i] == 1 and v[i] == 1 and vis[i] == 0):
                vis[i] = 1
                st.append(i)
    for i in range(v.shape[0]):
        if (vis[i] == 0 and v[i] == 1):
            return False
    return True


def check_connectivity(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch, avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity):
    
    mean_basis_weights = np.mean(basis_weights, axis = 0)
    top_bases = np.load("exist_bases.npy")
    top_bases = np.sort(top_bases)
    n_top_bases = len(top_bases)


    N = xrd.shape[0]
    M = basis_weights.shape[1]
    print("N = %d, M = %d"%(N, M))


    phase_dict = {}

    for i in range(N):
        tmp = []
        for j in range(n_top_bases):
            if (basis_weights[i][top_bases[j]] > 1e-6):
                tmp.append(top_bases[j])
        for j in range(1, 1<<len(tmp)):
            ttmp = []
            for k in range(len(tmp)):
                if ((j>>k)&1 == 1):
                    ttmp.append(tmp[k])
            phase_dict[tuple(ttmp)] = 1



    edges = np.load(FLAGS.edges_dir)
    sth_is_wrong = 0
    for key in phase_dict.keys():
        v = np.zeros(N)
        V = []
        for i in range(N):
            flag = 1
            for j in key:
                if (basis_weights[i][j] < 1e-6):
                    flag = 0
            v[i] = flag
            if (v[i] == 1):
                V.append(i)

        
        if (np.sum(v) > 0):
            
            if (is_connected(edges, v)):
                pass
            else:
                print("NOT CONNECTED!")
                print(key)
                sth_is_wrong += 1

                if ("visual" in sys.argv):
                    f, axes = plt.subplots(1, 1, figsize = (12, 10))
                    coords = comp2Coords(composition)
                    X = coords[V, 0]
                    Y = coords[V, 1]
                    axes.scatter(coords[:, 0], coords[:, 1], s = (10.0)**2 , c = "black", alpha = 0.3)

                    axes.scatter(X, Y, s = (10.0)**2 , c = "red", alpha = 0.9)
                    for i in range(len(X)):
                        for j in range (i + 1, len(X)):
                            if (edges[V[i], V[j]] == 1):
                                axes.plot([X[i], X[j]],  [Y[i], Y[j]], c = "black", alpha= 0.5, zorder = -1)

                    plt.show()
                


    if (sth_is_wrong == 0):
        print("Perfect!")
    else:
        print("sth is wrong:", sth_is_wrong)

def fwrite(v, f):
    for i in range(v.shape[0]):
        c = ','
        if (i == v.shape[0] - 1):
            c = '\n'
        f.write("%.6f%c"%(v[i], c))

def sticks2xrd(basis, mu_shift, logvar, Q, rescale = True):

    mu_init = np.zeros(200)
    c = np.zeros(200)

    for (i, peak) in enumerate(basis):
        mu_init[i] = peak[0] #(peak[0] - ((80 + 15) / 2.0))/(80 - 15) * 2.0 * lim
        c[i] = peak[1]

    mu = mu_init * mu_shift
    eps = 1e-9
    X = np.copy(Q)
    x = np.zeros_like(X)
    for i in range(len(basis)):
        x += np.exp( - ((X - mu[i])**2)/(2.0 * np.exp(logvar) + eps) - logvar * 0.5) * c[i]  # xrd, bs, n_bases, n_spike

    if (rescale):
        x = x / (np.max(x) + eps)

    return x

def generate_solution(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch, avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity, intensity_shift):
    mean_basis_weights = np.mean(basis_weights, axis = 0)
    bases_name = np.load(FLAGS.bases_name_dir)
    top_bases = np.load("exist_bases.npy")
    top_bases = np.sort(top_bases)
    n_top_bases = len(top_bases)
    shift_status = np.load("shift_status.npy")
    max_shifts = np.load("max_shifts.npy")
    Q_idx = np.load(FLAGS.Q_idx_dir)
    Q = np.load(FLAGS.Q_dir)
    stick_bases = np.load(FLAGS.stick_bases_dir)
    bases_comp = np.load(FLAGS.bases_comp_dir) 
    bases_edge = np.load(FLAGS.bases_edge_dir)

    f = open("solu.txt", "w")

    f.write("// Number of phases\n")
    f.write("K=%d\n"%n_top_bases)
    for i in range(n_top_bases):
        f.write("BName%d="%(i+1) + bases_name[top_bases[i]][:-4])
        for j in range(bases_name.shape[0]):
            if (bases_edge[top_bases[i]][j] == 1):
                f.write(","+bases_name[j][:-4])
        f.write("\n")


    f.write("\n")

    f.write("// Phase patterns (basis)\n")
    f.write("Q=")
    fwrite(Q, f)
    for i in range(n_top_bases):
        f.write("B%d="%(i + 1))
        x = sticks2xrd(stick_bases[top_bases[i]], 1.0, -4.75, Q)
        fwrite(x, f)

    f.write("\n// Per-phase model for each sample\n")
    for i in range(basis_weights.shape[0]):
        x_prime = np.sum(decomp[i], axis = 0)
        x_max = np.max(x_prime)
        for j in range(n_top_bases):
            f.write("R%d_%d="%(i + 1, j + 1))
            b = top_bases[j]
            if (basis_weights[i][b] > 1e-6):
                basis = decomp[i][b] / (x_max + 1e-9)
                print(basis.shape)
                fwrite(basis, f)
            else:
                basis = np.zeros_like(decomp[i][b])
                fwrite(basis, f)

    f.write("\n// Phase concentrations at each sample\n")
    for i in range(basis_weights.shape[0]):
        f.write("C%d="%(i + 1))
        fwrite(basis_weights[i][top_bases], f)

    f.write("\n// Phase shifts at each sample\n")
    for i in range(basis_weights.shape[0]):
        f.write("S%d="%(i + 1))
        fwrite(mu_shift[i][top_bases], f)

    f.write("\n// intensity shifts at each sample (to make every library stick patterns have the same number of sticks, we added some 0 intensity sticks for each library phase. Therefore, you can ignore the intensity_shift for those padding sticks)\n")
    for i in range(basis_weights.shape[0]):
        for j in range(n_top_bases):
            f.write("IS%d_%d="%(i + 1, j + 1))
            b = top_bases[j]
            fwrite(intensity_shift[i][b], f)


    f.write("\n// Per-sample contribution (L1 loss)\n")
    f.write("L=")
    for i in range(basis_weights.shape[0]):
        x = xrd[i][Q_idx] / (np.max(xrd[i][Q_idx]) + 1e-9)
        x_prime = np.sum(decomp[i], axis = 0)
        x_prime /= (1e-9 + np.max(x_prime))
        c = ","
        if (i == basis_weights.shape[0] - 1):
            c = "\n"
        f.write("%.6f%c"%(np.sum(np.abs(x - x_prime)), c))

    f.write("L_proportion=")
    for i in range(basis_weights.shape[0]):
        x = xrd[i][Q_idx] / (np.max(xrd[i][Q_idx]) + 1e-9)
        x_prime = np.sum(decomp[i], axis = 0)
        x_prime /= (1e-9 + np.max(x_prime))

        c = ","
        if (i == basis_weights.shape[0] - 1):
            c = "\n"
        #print(np.sum(np.abs(x - x_prime)), np.sum(np.abs(x)))
        #plt.plot(Q, x)
        #plt.plot(Q, x_prime)
        #plt.show()
        f.write("%.6f%c"%(np.sum(np.abs(x - x_prime))/np.sum(np.abs(x)), c))


    f.close()

    f = open("inst.txt", "w")
    f.write("//Integrated counts data\n")
    f.write("Q=")
    fwrite(Q, f)
    for i in range(basis_weights.shape[0]):
        f.write("I%d="%(i + 1))
        fwrite(xrd[i][Q_idx] / (np.max(xrd[i]) + 1e-9), f)
    f.close()


def main(_):

    print('reading npy...')
    #np.random.seed(19950420) # set the random seed of numpy 
    data, batches = get_data.get_data() #XRD sources and batches
    train_idx = np.arange(batches.shape[0]) #load the indices of the training set

    xrd = get_data.get_xrd_mat()
    composition = get_data.get_comp()
    degree_of_freedom = get_data.get_degree_of_freedom(np.arange(FLAGS.testing_size))
    #max_peak = np.max(xrd, axis=1)
    #plt.hist(max_peak)
    #plt.show()
    #visual_bases_sol(bases_sol)

    one_epoch_iter = train_idx.shape[0] # compute the number of iterations in each epoch

    print('reading completed')

    # config the tensorflow
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    print('showing the parameters...\n')

    for key in FLAGS:
        value = FLAGS[key].value
        print("%s\t%s"%(key, value))
    print("\n")


    print('building network...')

    #building the model 
    hg = model.MODEL(is_training=True)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, (1.0/FLAGS.lr_decay_times)*(FLAGS.max_epoch*one_epoch_iter), FLAGS.lr_decay_ratio, staircase=True)

    #log the learning rate 
    tf.summary.scalar('learning_rate', learning_rate)

    #use the Adam optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate)

    #set training update ops/backpropagation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(hg.optimizer_loss, global_step = global_step)

    #merged_summary = tf.summary.merge_all() # gather all summary nodes together
    #summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph) #initialize the summary writer

    Vars = tf.global_variables()
    Vars_filtered = []
    for v in Vars:
        if (not ("Adam" in v.name and "spike_shift" in v.name)):
            Vars_filtered.append(v)

    saver = tf.train.Saver(max_to_keep=None, var_list = Vars_filtered)
    
    saver.restore(sess,FLAGS.checkpoint_path)

    print('restoring from '+FLAGS.checkpoint_path)


    print('Testing...')
    N = FLAGS.testing_size
    M = FLAGS.n_bases

    st_basis_weights = []
    st_avg_recon_loss = []
    st_recon_loss = []
    st_JS_dis_batch = []
    st_L2_dis_batch = []
    st_decomp = [] 
    st_xrd_prime  = []
    st_mu = []
    st_mu_shift = []
    st_logvar = []
    st_intensity = []
    st_gibbs_loss_batch = []
    st_alloy_loss_batch = []
    st_comp_prime = []
    st_comp_loss_batch = []
    st_intensity_shift = []
    for j in range(N):
        #if (j%10 == 0):
        #   print("%.2f%c"%(j*100.0/N, '%'))
        idx = [j]
        feed_dict={}
        feed_dict[hg.input_feature] = get_data.get_feature(data, idx) # get the FEATURE features 
        feed_dict[hg.input_xrd] = get_data.get_xrd(data, idx) 
        
        if ("refine" in sys.argv):
            if ("new" in sys.argv):
                feed_dict[hg.input_indicator] = get_data.get_refined_sample_indicator(idx, 2)
                feed_dict[hg.shift_indicator] = get_data.get_refined_shift_indicator(idx, 2)
            else:
                feed_dict[hg.input_indicator] = get_data.get_refined_sample_indicator(idx)
                feed_dict[hg.shift_indicator] = get_data.get_refined_shift_indicator(idx)
        else:
            feed_dict[hg.input_indicator] = get_data.get_indicator(idx)
            feed_dict[hg.shift_indicator] = np.zeros((len(idx), 1)) + 1


        feed_dict[hg.degree_of_freedom] = get_data.get_degree_of_freedom(idx)
        feed_dict[hg.keep_prob] = FLAGS.keep_prob
        feed_dict[hg.epoch] = 10.0

        noise, tmp_JS_dis_batch, tmp_L2_dis_batch, tmp_comp_loss_batch, tmp_comp_prime, tmp_gibbs_loss_batch, tmp_alloy_loss_batch, tmp_basis_weights, \
        tmp_avg_recond_loss, tmp_recon_loss, tmp_decomp, tmp_xrd_prime, tmp_mu, tmp_mu_shift, tmp_logvar, tmp_intensity, tmp_intensity_shift = \
        sess.run([hg.noise, hg.JS_dis_batch, hg.L2_dis_batch, hg.comp_loss_batch, hg.comp_prime, hg.gibbs_loss_batch, hg.alloy_loss_batch, hg.weights,\
         hg.recon_loss, hg.recon_loss_batch, hg.decomp, hg.xrd_prime, hg.mu, hg.mu_shift, hg.logvar, hg.intensity, hg.intensity_shift], feed_dict)

        st_JS_dis_batch.append(tmp_JS_dis_batch)
        st_L2_dis_batch.append(tmp_L2_dis_batch)
        st_comp_loss_batch.append(tmp_comp_loss_batch)
        st_comp_prime.append(tmp_comp_prime)
        st_gibbs_loss_batch.append(tmp_gibbs_loss_batch)
        st_alloy_loss_batch.append(tmp_alloy_loss_batch)
        st_basis_weights.append(tmp_basis_weights)
        st_avg_recon_loss.append(tmp_avg_recond_loss)
        st_recon_loss.append(tmp_recon_loss)
        st_decomp.append(tmp_decomp)
        st_xrd_prime.append(tmp_xrd_prime)
        st_mu.append(tmp_mu)
        st_mu_shift.append(tmp_mu_shift)
        st_logvar.append(tmp_logvar)
        st_intensity.append(tmp_intensity)
        st_intensity_shift.append(tmp_intensity_shift)

    JS_dis_batch = np.concatenate(st_JS_dis_batch, axis = 0)
    L2_dis_batch = np.concatenate(st_L2_dis_batch, axis = 0)
    comp_loss_batch = np.concatenate(st_comp_loss_batch, axis = 0)
    comp_prime = np.concatenate(st_comp_prime, axis = 0)
    gibbs_loss_batch = np.concatenate(st_gibbs_loss_batch, axis = 0)
    alloy_loss_batch = np.concatenate(st_alloy_loss_batch, axis = 0)
    basis_weights = np.concatenate(st_basis_weights, axis = 0)
    avg_recon_loss = np.mean(st_avg_recon_loss)
    recon_loss = np.concatenate(st_recon_loss, axis = 0)
    decomp = np.concatenate(st_decomp, axis = 0)
    xrd_prime  = np.concatenate(st_xrd_prime, axis = 0)
    mu = np.concatenate(st_mu, axis = 0)
    mu_shift = np.concatenate(st_mu_shift, axis = 0)
    logvar = np.concatenate(st_logvar, axis = 0)
    intensity = np.concatenate(st_intensity, axis = 0)
    intensity_shift = np.concatenate(st_intensity_shift, axis = 0)


    if ("ternary" in sys.argv):
        plot_ternary(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch,\
     avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity)


    if ("recon" in sys.argv):
        plot_recon(xrd, JS_dis_batch, L2_dis_batch, degree_of_freedom, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch, alloy_loss_batch, basis_weights,\
     avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity)

    if ("check" in sys.argv):
        check_connectivity(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch,\
     avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity)

    if ("solu" in sys.argv):
        generate_solution(xrd, degree_of_freedom, basis_weights, comp_loss_batch, composition, comp_prime, noise, gibbs_loss_batch,\
     avg_recon_loss, recon_loss, decomp, xrd_prime, mu, mu_shift, logvar, intensity, intensity_shift)

    ######################################################

    ######################################################

    
if __name__=='__main__':
    tf.app.run()



