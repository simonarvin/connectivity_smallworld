import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import cycler
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
import pandas as pd
import ppscore as pps
from dominance_analysis import Dominance

import pathlib
import sys


data_name = f"sw_kuramoto_stability"
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"
data_path = f"{base_path}/{data_name}.json"

crop_length = 250

matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams.update({'errorbar.capsize': 1.5})

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def logifunc(x, A, B, C, D):
    """
    for scipy.curve_fit: logistic regression
    https://se.mathworks.com/matlabcentral/fileexchange/38122-four-parameters-logistic-regression-there-and-back-again
    """
    return ((A-D)/(1.0+((x/C)**(B))) + D)

def fit(x, y, real = False): #fitting function
    betas = np.logspace(np.log10(0.0005), np.log10(10), 70)
    p0 = [np.amax(y), np.median(x), 2, np.amin(y)] # initial guess for scipy.curve_fit
    popt, pcov = curve_fit(logifunc, x, y, p0, method = "trf")
    if real: #excludes error margins. Print fitted parameters
        print(f"fitted parameters:\n{popt}")

    return betas, logifunc(betas, *popt)

class Analyser:
    def __init__(self):
        self.lines = []
        with open(data_path) as file:
            lines = file.readlines()
            for line in lines: # load data
                js = json.loads(line)
                self.lines.append(js)

    def plot(self):
        fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize=(2 * 4, 2*2), sharey=True)
        fig2, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize=(2, 2))


        longrange_probabilities = np.logspace(np.log10(0.0001), np.log10(1), 9)
        coherences = np.flip(np.linspace(0, 1, 10))
        shortranges = [5, 25, 50]

        heatmap = np.zeros((len(longrange_probabilities), len(coherences)), dtype=np.float64)
        multipliers = [heatmap.copy(), heatmap.copy(),heatmap.copy()]
        heatmaps = [heatmap.copy(), heatmap.copy(),heatmap.copy()] #* len(shortranges)

        #fig_longrange_synchrony, axes_longrange_synchrony = plt.subplots(nrows = 1, ncols = 3, figsize=(2.7 * 3,3))

        #fig_longrange_synchrony.canvas.manager.set_window_title('Long-range synchrony')
        #fig_coherences.canvas.manager.set_window_title('Sample phase coherences')

        #coherence = [entry["phase_cohs"] for entry in self.lines]

        for entry in self.lines:
            #print(entry["longrange_probability"])
            #print(shortranges.index(entry["shortrange_pernode"]))
            heatmap = heatmaps[shortranges.index(entry["shortrange_pernode"])]
            multiplier = multipliers[shortranges.index(entry["shortrange_pernode"])]
            phase_coherence = np.array(entry["phase_cohs"]).T

            inbound_diff2 = np.diff(np.diff(phase_coherence, axis = 0), axis=0)
            inbound_diff1 = np.diff(phase_coherence, axis = 0)
            span = np.mean(np.amin(inbound_diff1,axis=0) - np.amax(inbound_diff1,axis=0))
            duration = np.mean(np.argmin(inbound_diff1,axis=0) - np.argmax(inbound_diff1,axis=0))
            std_diff2 = np.mean(np.std(inbound_diff2, axis=0))

            crop_ = phase_coherence[:crop_length, :]

            crop = -1
            std_window =   (np.abs(np.mean(crop_[crop:], axis = 1) - entry["coherence"]))#np.sqrt(np.sum((crop_[-25:] - entry["coherence"])**2, axis=0)/(crop_.shape[0] - 1))
            idx = (np.abs(coherences - np.mean(crop_[crop:], axis = 1))).argmin()
            multiplier[np.where(longrange_probabilities == entry["longrange_probability"]), idx] += -np.mean(np.abs(entry["coherence"] - np.mean(crop_[crop:], axis = 1)))
            ####std_window = (np.sqrt(np.sum((crop_[crop:] - entry["coherence"])**2, axis = 1)/crop_.shape[1]))#

            ####(np.sqrt(np.sum((crop_[-25:] - entry["coherence"])**2, axis = 1)/crop_.shape[1]))#
            #std_diff1 = np.mean(std_window)
            std_diff1 = np.mean(std_window) #* np.std(std_window)
            #std_diff1 = np.mean(std_window * 2*(sigmoid(np.flip(np.arange(std_window.shape[0]))/std_window.shape[0])-.5)[np.newaxis].T)#np.mean(np.std(inbound_diff1, axis=0))
            #print(sigmoid(np.flip(np.arange(std_window.shape[0]))/std_window.shape[0]))
            heatmap[np.where(longrange_probabilities == entry["longrange_probability"]), np.where(coherences == entry["coherence"])] = std_diff1#duration

            #print("span: ", span)
            #print("duration: ", duration)
            #print("std diff2: ", std_diff2)

                #print(phase_coherence.shape)
                #axs.plot(phase_coherence)
                #axs.plot(inbound_diff2)
                #axs.plot(inbound_diff1)
                #break


        comb_heat = np.hstack(heatmaps)
        diff1 = heatmaps[0].T - heatmaps[1].T
        diff2 = heatmaps[1].T - heatmaps[2].T
        diff3 = heatmaps[0].T - heatmaps[2].T
        comb_diff = np.hstack((diff1, diff2, diff3))

        for i, ax in enumerate(axs.reshape(-1)):
            if i < 3:
                smooth = heatmaps[i].T #gaussian_filter(heatmap.T, sigma=1)

                ax.set_title(f"$H = {shortranges[i]}$", fontsize = 10)

                c=ax.pcolor(longrange_probabilities, coherences, smooth, vmin = np.min(comb_heat), vmax=np.max(comb_heat),cmap="viridis")
                print(np.min(comb_heat),np.max(comb_heat))
                #ax.imshow(smooth,  vmin = np.min(comb_heat), vmax=np.max(comb_heat), extent=[longrange_probabilities[0], 1, 0, 1],cmap='viridis', interpolation='none')
                #vmin=.0, vmax=.3,
                #ax.set_xscale("log")
                #continue
                ax.set_yticks([0, .5, 1])
                #ax.set_yticklabels(np.round(coherences, 2)[::3])
                #continue
                #if i == 0:
                #xticks = [0, 4, 7]
            elif i == 3:
                cb=fig.colorbar(c, ax=ax, ticks=[np.min(comb_heat),np.max(comb_heat)])
                #cb.ax.set_yticks([np.min(comb_heat),np.max(comb_heat)])

            elif i == 4:

                ax.pcolor(longrange_probabilities, coherences, diff1, vmin = np.min(comb_diff), vmax=np.max(comb_diff))
                #ax.imshow(diff, vmin = np.min(diff), vmax=np.max(diff),extent=[longrange_probabilities[0], 1, 0, 1],cmap='viridis', interpolation='none')

            elif i == 5:

                ax.pcolor(longrange_probabilities, coherences, diff2, vmin = np.min(comb_diff), vmax=np.max(comb_diff))
            elif i == 6:

                ax.pcolor(longrange_probabilities, coherences, diff3, vmin = np.min(comb_diff), vmax=np.max(comb_diff))
            #ax.set_xticks(longrange_probabilities*10)
            #ax.set_xticklabels([])

            ax.set_xscale("log")

        print(np.sum(diff3, axis = 1).shape)
        #skal korrigeres for log skala
        #print(diff3/longrange_probabilities)
        threshold = np.mean(diff3)
        #print(diff3[diff3 < threshold])
        norm_diff3 = diff3 #* longrange_probabilities
        #norm_diff3[np.abs(diff3) < threshold] = 0
        #np.sum(norm_diff3, axis = 1)
        norm_diff3 = multipliers[2].T - multipliers[0].T
        ax2.errorbar(coherences, np.mean(norm_diff3, axis = 1), yerr=stats.sem(norm_diff3, axis = 1), fmt=".-", color = "#E33237")
        ax2.set_yticks([3, 0, -1])
        ax2.set_xticks([0, .25, .5, .75, 1])

        #axs.set_xscale("log")
        #axs.set_xlim(0,1)
        #axs.set_xticklabels(longrange_probabilities)
        #axs[1].set_ylim(0,1)
        plt.tight_layout()
        plt.show()

a = Analyser()
a.plot()
