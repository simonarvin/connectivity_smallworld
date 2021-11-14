import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import cycler
from scipy import stats
from scipy.ndimage.filters import gaussian_filter

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
        attractionmaps = [heatmap.copy(), heatmap.copy(),heatmap.copy()]
        heatmaps = [heatmap.copy(), heatmap.copy(),heatmap.copy()] #* len(shortranges)


        for entry in self.lines:

            heatmap = heatmaps[shortranges.index(entry["shortrange_pernode"])]
            attractionmap = attractionmaps[shortranges.index(entry["shortrange_pernode"])]
            phase_coherence = np.array(entry["phase_cohs"]).T

            crop_ = phase_coherence[:crop_length, :]
            crop = -1

            deviation =   (np.abs(np.mean(crop_[crop:], axis = 1) - entry["coherence"]))
            heatmap[np.where(longrange_probabilities == entry["longrange_probability"]), np.where(coherences == entry["coherence"])] = np.mean(deviation)

            end_regime = (np.abs(coherences - np.mean(crop_[crop:], axis = 1))).argmin()
            attractionmap[np.where(longrange_probabilities == entry["longrange_probability"]), end_regime] += -np.mean(np.abs(entry["coherence"] - np.mean(crop_[crop:], axis = 1)))


        comb_heat = np.hstack(heatmaps)
        diff1 = heatmaps[0].T - heatmaps[1].T #H = 10 vs H = 50
        diff2 = heatmaps[1].T - heatmaps[2].T #H = 50 vs H = 100
        diff3 = heatmaps[0].T - heatmaps[2].T #H = 10 vs H = 100
        comb_diff = np.hstack((diff1, diff2, diff3))

        for i, ax in enumerate(axs.reshape(-1)):
            if i < 3:
                smooth = heatmaps[i].T #gaussian_filter(heatmap.T, sigma=1)

                ax.set_title(f"$H = {shortranges[i]*2}$", fontsize = 10)

                c=ax.pcolor(longrange_probabilities, coherences, smooth, vmin = np.min(comb_heat), vmax = np.max(comb_heat),cmap = "viridis")

                ax.set_yticks([0, .5, 1])

            elif i == 3:
                #colorbar
                cb = fig.colorbar(c, ax = ax, ticks = [np.min(comb_heat), np.max(comb_heat)])


            elif i == 4:
                #H = 10 vs H = 50
                ax.pcolor(longrange_probabilities, coherences, diff1, vmin = np.min(comb_diff), vmax=np.max(comb_diff))
                #ax.imshow(diff, vmin = np.min(diff), vmax=np.max(diff),extent=[longrange_probabilities[0], 1, 0, 1],cmap='viridis', interpolation='none')

            elif i == 5:
                #H = 50 vs H = 100
                ax.pcolor(longrange_probabilities, coherences, diff2, vmin = np.min(comb_diff), vmax=np.max(comb_diff))
            elif i == 6:
                #H = 10 vs H = 100
                ax.pcolor(longrange_probabilities, coherences, diff3, vmin = np.min(comb_diff), vmax=np.max(comb_diff))


            ax.set_xscale("log")

        
        attraction_diff = attractionmaps[2].T - attractionmaps[0].T
        ax2.errorbar(coherences, np.mean(attraction_diff, axis = 1), yerr=stats.sem(attraction_diff, axis = 1), fmt=".-", color = "#E33237")
        ax2.set_yticks([3, 0, -1])
        ax2.set_xticks([0, .25, .5, .75, 1])


        plt.tight_layout()
        plt.show()

a = Analyser()
a.plot()
