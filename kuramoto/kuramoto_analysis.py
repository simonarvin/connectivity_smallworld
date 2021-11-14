import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import cycler

from scipy.optimize import curve_fit
import pandas as pd
import ppscore as pps
from dominance_analysis import Dominance

import pathlib
import sys

try:
    if "." in sys.argv[1]:
        coupling = float(sys.argv[1])
    else:
        coupling = int(sys.argv[1])
except:
    coupling = 3 #Kuramoto's coupling parameter. Data-set includes couplings = [1, 2, 2.5, 3, 4]

print(f"coupling: {coupling}")
data_name = f"sw_kuramoto_coupling{coupling}"
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"
data_path = f"{base_path}/{data_name}.json"
colours = ["#E33237", "#1C9E77", "#2E3192"]

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
        fig_coherences, axes_coherences = plt.subplots(nrows = 3, ncols = 1, figsize=(4, 1 * 3))
        fig_longrange_synchrony, axes_longrange_synchrony = plt.subplots(nrows = 1, ncols = 3, figsize=(2.7 * 3,3))

        fig_longrange_synchrony.canvas.manager.set_window_title('Long-range synchrony')
        fig_coherences.canvas.manager.set_window_title('Sample phase coherences')

        coherence = [[np.mean(coh[-1000:]) for coh in entry["phase_cohs"]] for entry in self.lines]
        raw_coherence = [[coh[-1500:] for coh in entry["phase_cohs"]] for entry in self.lines]
        coherence_std = [np.std(coh) for coh in coherence] #OBS, STD

        #longs = [np.mean(entry["longrange_connections_arr"])/entry["nodes"] for entry in self.lines]
        longs = [np.array(entry["longrange_connections_arr"])/entry["nodes"] for entry in self.lines]
        shorts = [entry["shortrange_pernode"] for entry in self.lines]

        #check dominance of variables g, h and r onto the metastability r_std
        stack = np.mean(np.array(longs),axis=1), np.array(shorts),  np.mean(np.array(coherence), axis=1), np.array(coherence_std),
        dominance_matrix = np.vstack(stack).T
        pandas_matrix = pd.DataFrame(dominance_matrix, columns = ['g', 'h', 'r', 'r_std'])

        dominance_regression=Dominance(data = pandas_matrix,target = 'r_std',objective=1)
        incr_variable_rsquare = dominance_regression.incremental_rsquare()
        print(f"\ndominance analysis:\n{incr_variable_rsquare}\n")

        lower_index = 0
        color_index = 0

        for index, short in enumerate(shorts):

            ax = axes_longrange_synchrony[color_index]
            ax2 = axes_coherences[color_index]

            if color_index == 0:
                cols = plt.cm.PuRd(np.linspace(0, 1, 6))
            elif color_index == 1:
                cols = plt.cm.GnBu(np.linspace(0, .7, 6))
            else:
                cols = plt.cm.BuPu(np.linspace(0, 1, 6))
            ax2.set_prop_cycle(cycler.cycler('color', cols))

            try:
                if short != shorts[index + 1]:

                    index += 1

                    #compute mean curve
                    x, y = np.mean(longs, axis = 1)[lower_index:index], np.mean(coherence, axis = 1)[lower_index:index]

                    #fit curve
                    x_, y_ = fit(x, y, real = True)

                    #fit error margins, std
                    _, ey1_ = fit(x, y + coherence_std[lower_index:index])
                    _, ey2_ = fit(x, y - coherence_std[lower_index:index])

                    #plot error margin, std
                    ax.fill_between(x_, ey2_, ey1_, color=colours[color_index], alpha=0.2, zorder=-1, lw = 0)

                    #plot fitted curve
                    ax.plot(x_, y_, c=colours[color_index], lw=1)

                    #plot means +- std
                    ax.errorbar(x, y, yerr = coherence_std[lower_index:index], fmt='o', c=colours[color_index],markersize=2, lw = 1, capsize = 1.5)

                    longs_scope = np.array(longs)[lower_index:index,:]
                    coherence_scope =  np.array(coherence)[lower_index:index,:]
                    raw_coherence_scope =  np.array(raw_coherence)[lower_index:index,:,:]

                    sample_trials = 6 #number of samples to plot
                    for sample in range(sample_trials):
                        ax2.plot(np.arange(1500), raw_coherence_scope[3, sample, :], lw = 1)

                    #compute the predictive score of long-range connectivity g on network synchrony r
                    df = pd.DataFrame()
                    df["g"] = longs_scope.flatten() #long-range connectivity
                    df["r"] = coherence_scope.flatten() #y
                    predictive_score = pps.score(df, "g", "r")

                    print(f"\npredictive score:\n{predictive_score}\n")

                    lower_index = index
                    color_index += 1

            except: #catch last plot:

                x, y = np.mean(longs, axis = 1)[lower_index:], np.mean(coherence, axis = 1)[lower_index:]

                try:
                    x_, y_ = fit(x, y, real = True)
                    _, ey1_ = fit(x, y + coherence_std[lower_index:])
                    _, ey2_ = fit(x, y - coherence_std[lower_index:])
                except TypeError:
                    continue

                ax.fill_between(x_, ey2_, ey1_, color=colours[color_index],alpha=0.2, zorder=-1, lw=0)
                ax.plot(x_, y_, c=colours[color_index], lw=1)
                ax.errorbar(x, y, yerr = coherence_std[lower_index:], fmt='o', c=colours[color_index],markersize=2, lw = 1, capsize = 1.5)

                longs_scope = np.array(longs)[lower_index:,:]
                coherence_scope =  np.array(coherence)[lower_index:,:]
                raw_coherence_scope =  np.array(raw_coherence)[lower_index:,:,:]

                sample_trials = 6 #number of samples to plot
                for sample in range(sample_trials):
                    ax2.plot(np.arange(1500), raw_coherence_scope[3, sample, :], lw = 1)

                #compute the predictive score of long-range connectivity g on network synchrony r
                df = pd.DataFrame()
                df["g"] = longs_scope.flatten() #long-range connectivity
                df["r"] = coherence_scope.flatten() #y
                predictive_score = pps.score(df, "g", "r")

                print(f"\npredictive score:\n{predictive_score}\n")



        #set plot layout, interface
        for index, ax in enumerate(axes_longrange_synchrony):
            ax.set_xscale('log')
            ax.set_yticks([0, .25, .5, .75, 1])

            locmaj = matplotlib.ticker.LogLocator(base=10,numticks=4)
            ax.xaxis.set_major_locator(locmaj)
            locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if index == 0:
                ax.set_xlabel("long-range connectivity,\np [wires/node]")
                ax.set_ylabel("global phase coherence, r")
            else:
                ax.set_yticklabels([])

            ax.set_ylim(0,1)
            axes_coherences[index].set_ylim(-.025, .7)
            axes_coherences[index].set_yticklabels([])
            axes_coherences[index].set_xticklabels([])
            axes_coherences[index].set_yticks([0, .7])

        plt.tight_layout()
        plt.show()

a = Analyser()
a.plot()
