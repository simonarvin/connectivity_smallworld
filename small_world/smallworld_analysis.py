import json
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy import stats
import pandas as pd
from dominance_analysis import Dominance_Datasets
from dominance_analysis import Dominance

import pathlib
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"

matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams.update({'errorbar.capsize': 1.5})

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)

np.set_printoptions(suppress=True)
colours = ["#E33237","#1B9E77","#2E3192"]
grey = "#E6E6E6"

def dominance(data, split_indices):

    entries = data["longrange_per_node"].shape[1]
    k_ = np.zeros(entries)
    p_ = k_.copy()

    for i in range(entries):

        df = {
        "omega": data["omegas"][:,i],
        "k" : data["shortrange_pernode"],
        "p" : data["longrange_per_node"][:,i]
        }

        # crop to p_pernode = 10
        df = {
        "omega": data["omegas"][:split_indices["p_split"],i],
        "k" : data["shortrange_pernode"][:split_indices["p_split"]],
        "p" : data["longrange_per_node"][:split_indices["p_split"],i]
        }

        df = pd.DataFrame(df)

        dom = dominance_analysis(df)
        k_[i] = dom["k"]
        p_[i] = dom["p"]

    print(f"k: mean: {np.mean(k_)} SEM: {stats.sem(k_)}")
    print(f"p: mean: {np.mean(p_)} SEM: {stats.sem(p_)}")
    print(f"t-test {stats.ttest_ind(k_, p_)}")

    fig, ax = plt.subplots(1, 1, figsize = (.8, 1.2))


    ax.bar(["k", "p"], [np.mean(k_), np.mean(p_)], yerr = [stats.sem(k_), stats.sem(p_)], width=.5, color = ["#666666", "#E33237"], linewidth = 1, edgecolor="black")
    ax.tick_params(which="both", axis ="both", width=1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    plt.show()
    #print(omegas.reshape(-1).shape, shortrange_pernode.shape)

    #df = {
    #"omega": np.mean(omegas, axis = 1),
    #"k" : data["shortrange_pernode"],
    #"p" : np.mean(longrange_per_node, axis = 1)
    #}

    #return pd.DataFrame(df)

def dominance_analysis(df):
    dominance_regression = Dominance(data=df, target='omega',objective=1)
    d = dominance_regression.incremental_rsquare()
    return d


def split_arr(l, n:int):
    n = int(n)
    return np.array([l[i:i + n] for i in range(0, len(l), n)])

nodes = 1000


def load_entries():
    print("\n")
    try:
        file_path = sys.argv[1]
        file_name = os.path.basename(file_path)
    except:
        file_path = f"{base_path}/sw_dataset.json"
        file_name = "sw_dataset.json"
    print(f"opening file {file_name}\n")
    with open(file_path) as file:
        lines = file.readlines()
        entries = np.zeros(len(lines), dtype=dict)
        for i, line in enumerate(lines):
            entries[i] = json.loads(line)
        print(f"{file_name} succesfully loaded\n")

    return entries

def data_split(entries):

    shortrange_pernode = np.array([d["shortrange_pernode"] for d in entries])
    betas = np.array([d["beta"] for d in entries])
    omegas = np.array([d["omega_list"] for d in entries])
    longrange_per_node = np.array([d["Ns"] for d in entries])/nodes #[:,:,2] legacy
    ppernode = np.array([d["p_pernode"] for d in entries])

    data = {
    "shortrange_pernode" : shortrange_pernode,
    "betas" : betas,
    "omegas": omegas,
    "longrange_per_node" : longrange_per_node
    }

    print("data succesfully parsed")

    return data

def get_split_indices():
    """
    architecture
    for _, ppernode in enumerate(p_pernode):
        for _, shortpernode in enumerate(shortrange_pernode):
            for _, beta in enumerate(betas):
                [[for i, _ in enumerate(randint):]]
    """
    #betas = np.logspace(np.log10(0.0001), np.log10(1), 15)
    betas = np.logspace(np.log10(0.0001), np.log10(1), 12)
    np.append(betas, [1.5, 2])

    #shortrange_pernode = [4, 5, 10, 20,30, 40,50, 60,70, 80,90, 100]#np.arange(lower, upper, 5)
    shortrange_pernode = [5, 10, 30, 50, 70, 90, 100]

    p_pernode = [10, 50, 200]
    #p_pernode = [10]

    randint_len = 100

    split_indices = {
    "short_len" : len(shortrange_pernode),
    "short_split" : len(betas),
    "p_split" : len(shortrange_pernode) * len(betas),
    "samples" : randint_len,
    "beta" : betas,
    "p_len" : len(p_pernode)
    }

    print("split indices computed")

    return split_indices

def robustness(data, split_indices):
    omega_split = split_arr(data["omegas"], split_indices["p_split"])
    omega_split_ = []
    for i, split in enumerate(omega_split):
        omega_split_.append(split_arr(split, split_indices["short_split"]))

    omega_split_ = np.array(omega_split_)
    #todo
    #du har nu (n, 12, 15, 100) hvor n er antallet af long-range densiteter.
    #du skal nu loope igennem n (axis = 0) og plotte grafer. Brug beta for at normalisere

    fig, axs = plt.subplots(1, 3, figsize = (6, 2))
    fig_small, axs_small = plt.subplots(1, 3, figsize = (2.5, .8))
    curves = [1, 3, 6] # short-range curves
    AUC_collect = np.zeros(len(curves), dtype=np.ndarray)
    plot_critical_space(*axs)
    plot_critical_space(*axs_small)
    for i, n in enumerate(curves):
        #print(np.mean(omega_split_[:, n],axis=2).shape, split_indices["beta"].shape)
        #i = len(curves) - i - 1

        for q in range(split_indices["p_len"]):
        #    axs[i].plot(split_indices["beta"], np.mean(omega_split_[:, n],axis=2)[q])


            axs[i].errorbar(split_indices["beta"], np.mean(omega_split_[:, n],axis=2)[q], yerr = stats.sem(omega_split_[:, n],axis=2)[q], c = colours[i],ls = "None")
            axs[i].plot(split_indices["beta"], np.mean(omega_split_[:, n],axis=2)[q], ".-", color = colours[i])

        axs_small[i].fill_between(split_indices["beta"], 0, np.mean(omega_split_[:, n],axis=2)[0] - np.mean(omega_split_[:, n],axis=2)[-1], facecolor=colours[i], alpha=.2)
        axs_small[i].plot(split_indices["beta"], np.mean(omega_split_[:, n],axis=2)[0] - np.mean(omega_split_[:, n],axis=2)[-1], ".-", color = colours[i])
        axs_small[i].errorbar(split_indices["beta"], np.mean(omega_split_[:, n],axis=2)[0] - np.mean(omega_split_[:, n],axis=2)[-1], yerr = stats.sem(omega_split_[0, n] - omega_split_[-1, n],axis=1), c = colours[i],ls = "None")

        #axs_small[i].plot(split_indices["beta"], np.mean(np.diff(np.mean(omega_split_[:, n], axis=2), axis=0), axis = 0), ".-", color = colours[i])
        #axs_small[i].errorbar(split_indices["beta"], np.mean(np.diff(np.mean(omega_split_[:, n], axis=2), axis=0), axis = 0), yerr = stats.sem(np.diff(np.mean(omega_split_[:, n], axis=2), axis=0), axis=0), c = colours[i],ls = "None")
        print(omega_split_[:, n].shape)
        print(np.abs(np.diff(omega_split_[:, n], axis = 1)).shape)
        AUCs = np.sum(np.abs(omega_split_[:, n][0] - omega_split_[:, n][-1]), axis = 0)
        print("jjj",AUCs.shape, AUCs)
        #AUC_mean = np.mean(AUCs, axis = 1)
        #AUC_sem = stats.sem(AUCs, axis = 1)


        #print(f"AUC mean: {AUC_mean}; SEM: {AUC_sem}")
        AUC_collect[i] = AUCs


        axs_small[i].set_xscale("log")

        axs_small[i].set_ylim(-1, 1)
        axs_small[i].axhline(0,0,1, color = "black")
        axs_small[i].set_xticks([0.001, 0.1, 1])
        axs_small[i].set_xticklabels([])
        axs_small[i].set_yticklabels([])
        if i != 0:
            axs[i].set_yticklabels([])


        axs[i].set_xscale("log")
        axs[i].set_xticks([0.001, 0.1, 1])
        axs[i].set_ylim(-1, 1)

        axs[i].xaxis.set_minor_locator(locmin)
        axs[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[i].tick_params(which="both", axis ="both", width=1)
        axs_small[i].tick_params(which="both", axis ="both", width=1)

    #print("kkk",np.mean(AUC_collect[0]), np.mean(AUC_collect[-1]))
    #print(f"t-test {stats.ttest_ind(AUC_collect[0], AUC_collect[-1])}")
    from scipy.stats import ks_2samp
    for n1, a in enumerate(AUC_collect):
        for n2, b in enumerate(AUC_collect):
            if n1 < n2:
                print(n1, n2, ks_2samp(a, b))
    #print(stats.f_oneway(AUC_collect[0], AUC_collect[-1], axis = 1))

    plt.show()

    #print(omega_split.shape, omega_split_.shape)

def plot_critical_space(*axs, threshold = .5):
    for ax in axs:
        ax.axhline(threshold, 0, 1, linestyle = "--", color = grey,dashes=(4, 4))
        ax.axhline(-threshold, 0, 1, linestyle = "--", color = grey,dashes=(4, 4))

def omega_over_k(data, split_indices, plot = True):
    omega_extract = data["omegas"][:split_indices["p_split"]] #extract first long-range density = 10
    omega_split = split_arr(omega_extract, split_indices["short_split"]) #then split according to short-range density
    omega_split = np.swapaxes(omega_split, 0, 1)

    longrange_extract = data["longrange_per_node"][:split_indices["p_split"]] #do the same for long-range connectivity
    longrange_split = split_arr(longrange_extract, split_indices["short_split"]) #then split according to short-range density
    longrange_split = np.swapaxes(longrange_split, 0, 1)

    shortrange_extract = data["shortrange_pernode"][:split_indices["p_split"]] #do the same for long-range connectivity
    shortrange_split = split_arr(shortrange_extract, split_indices["short_split"]) #then split according to short-range density
    shortrange_split = np.swapaxes(shortrange_split, 0, 1)

    curves = [0, 4, 8, 11] #long-range curves
    #curves = [1, 3, 6]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (2.5, 3))
        plot_critical_space(ax)

        for curve in curves:
            print(f"long-range density: {np.mean(longrange_split[curve])}")

            ax.errorbar(shortrange_split[curve], np.mean(omega_split[curve], axis = 1), yerr = stats.sem(omega_split[curve], axis = 1), c = "#666666",ls = "None")
            ax.plot(shortrange_split[curve], np.mean(omega_split[curve], axis = 1), ".-", color = "#666666")

        ax.set_xscale("log")
        ax.tick_params(which="both", axis ="both", width=1)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([-1,-.5,0,.5,1])

        plt.show()

    return omega_split, shortrange_split, curves


def omega_over_p(data, split_indices, plot = True):
    omega_extract = data["omegas"][:split_indices["p_split"]] #extract first long-range density = 10
    omega_split = split_arr(omega_extract, split_indices["short_split"]) #then split according to short-range density

    longrange_extract = data["longrange_per_node"][:split_indices["p_split"]] #do the same for long-range connectivity
    longrange_split = split_arr(longrange_extract, split_indices["short_split"]) #then split according to short-range density
    curves = [1, 3, 6] #short-range curves
    #curves = [2, 6, 11]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (2.5, 3))
        vlines = [[.18,3.5], [0.04,11], [0.0085]] #critical thresholds
        plot_critical_space(ax)
        fig_small, ax_small = plt.subplots(1, 1, figsize = (2.5/3, .8))
        plot_critical_space(ax_small)

        for i, curve in enumerate(curves):
            print(f"short-range density: {data['shortrange_pernode'][curve * split_indices['short_split']]}")
            for vline in vlines[i]:
                ax.axvline(vline, 0, 1, color = colours[i], linestyle="--", alpha=.4, dashes=(4, 4))

            ax.errorbar(np.mean(longrange_split[curve], axis = 1), np.mean(omega_split[curve], axis = 1), xerr = stats.sem(longrange_split[curve], axis = 1), yerr = stats.sem(omega_split[curve], axis = 1), c = colours[i],ls = "None")

            ax.plot(np.mean(longrange_split[curve], axis = 1), np.mean(omega_split[curve], axis = 1), ".-", color = colours[i])
            ax_small.plot(np.mean(longrange_split[curve], axis = 1), np.mean(omega_split[curve], axis = 1), ".-", color = colours[i])

        ax.set_xscale("log")
        ax.tick_params(which="both", axis ="both", width=1)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([-1,-.5,0,.5,1])

        ax_small.tick_params(which="both", axis ="both", width=1)
        ax_small.xaxis.set_minor_locator(locmin)
        ax_small.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax_small.set_yticks([-1,-.5,0,.5,1])

        plt.show()

    return omega_split, longrange_split, curves


def mobility(data, split_indices):
    omega_split_p, longrange_split, curves_p = omega_over_p(data, split_indices, plot = False)
    omega_split_k, shortrange_split, curves_k = omega_over_k(data, split_indices, plot = False)

    fig, axs = plt.subplots(1, 2, figsize = (2.5 * 2, 2))
    beta_shape = np.resize(split_indices["beta"], omega_split_p[0].shape)

    fig_small, ax_small = plt.subplots(1, 1, figsize = (2.5/3, .8))
    vlines = [[.18,3.5], [0.04,11], [0.0085]] #critical thresholds

    for i, _ in enumerate(curves_p):
        for vline in vlines[i]:
            axs[0].axvline(vline, 0, 1, color = colours[i], linestyle = "--", alpha=.4, dashes=(4, 4))

    for i, curve in enumerate(curves_p):

        x, y = longrange_split[curve], omega_split_p[curve]
        a, b = np.diff(y, axis = 0), np.diff(x, axis = 0)
        y = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
        axs[0].errorbar(np.mean(x, axis = 1), np.mean(y, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(y, axis = 1), c = colours[i],ls = "None")
        axs[0].plot(np.mean(x, axis = 1), np.mean(y, axis = 1), ".-", color = colours[i])

        if i == 0:
            x_, y_ = x, y

        #axs[0].fill_between(np.mean(x, axis = 1), np.mean(y, axis = 1)-error, np.mean(y, axis = 1)+error, color = colours[i], alpha=.25,linewidth=0.0)


    yr_= np.divide(y_, y, out=np.zeros_like(y_), where=y!=0)
    ax_small.plot(np.mean(x, axis = 1), np.mean(yr_, axis = 1),".-", c=colours[0])
    ax_small.errorbar(np.mean(x, axis = 1), np.mean(yr_, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(yr_, axis = 1), c = colours[0],ls = "None")
    ax_small.axhline(1, 0, 1, color = "black")
    ax_small.fill_between(np.mean(x, axis = 1), 1, np.mean(yr_, axis=1), facecolor=colours[0], alpha=.2)
    #ax_small.plot([np.amin(x),np.amax(x)],[1, 1],color="black")

    y_ -= y
    y_ *= -1

    #vlines = [[.18,3.5], [0.04,11], [0.0085]] #critical thresholds
    axs[1].axvline((.0085 + .18)/2, 0, 1, color = grey)
    axs[1].axvline((.0085 + .18)/2 + np.std([0.0085,.18]), 0, 1, color = grey)
    axs[1].axvline((.0085 + .18)/2 - np.std([0.0085,.18]), 0, 1, color = grey)
    print(np.std([0.0085,.18]))
    axs[1].axvline(3.5, 0, 1, color = grey)
    axs[1].plot(np.mean(x, axis=1), np.mean(y_, axis=1), ".-", c=colours[0])
    axs[1].errorbar(np.mean(x, axis = 1), np.mean(y_, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(y_, axis = 1), c = colours[0],ls = "None")
    #axs[1].fill_between(np.mean(x, axis = 1), 0, np.mean(y_, axis = 1), facecolor=colours[0], alpha=.25)
    #error = stats.sem(y_, axis = 1)
    #axs[1].fill_between(np.mean(x, axis = 1), np.mean(y_, axis = 1)-error, np.mean(y_, axis = 1)+error, color = colours[0], alpha=.25,linewidth=0.0)


    for curve in curves_k:

        x, y = np.resize(shortrange_split[curve],omega_split_k[curve].shape), omega_split_k[curve]
        a, b = np.diff(y, axis = 0), np.diff(x, axis = 0)
        y = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
        #axs[0].errorbar(x, np.mean(y, axis = 1), yerr = stats.sem(y, axis = 1), c = "#666666",ls = "None")
        axs[0].plot(x, np.mean(y, axis = 1), ".-", color = "#666666")
        break


    ax_small.set_xscale("log")
    #ax_small.xaxis.set_minor_locator(locmin)
    #ax_small.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax_small.set_xticks([0.01,10])
    ax_small.tick_params(which="both", axis ="both", width=1)
    ax_small.set_ylim(0, 7)
    axs[1].set_xscale("log")


    axs[1].xaxis.set_minor_locator(locmin)
    axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


    axs[1].set_xticks([0.01,10])

    axs[1].set_yscale("symlog")
    axs[1].set_ylim([-1.5,20])
    axs[1].yaxis.set_minor_locator(locmin)
    axs[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axs[1].tick_params(which="both", axis ="both", width=1)


    axs[0].set_xscale("log")
    axs[0].set_yscale("symlog")
    axs[0].set_ylim(-.2, 40)
    axs[0].xaxis.set_minor_locator(locmin)
    axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axs[0].yaxis.set_minor_locator(locmin)
    axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axs[0].tick_params(which="both", axis ="both", width=1)

    plt.show()



entries = load_entries() #clean_entries()
data = data_split(entries)
split_indices = get_split_indices()

#robustness(data, split_indices)

mobility(data, split_indices)

omega_over_k(data, split_indices)
omega_over_p(data, split_indices)

dominance(data, split_indices)
#dominance_analysis(df)
#plot_shortrange(entries)
#data_handler(entries)
