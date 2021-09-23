import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
import matplotlib.ticker
import ppscore as pps
import pandas as pd
import cycler

import pathlib

coupling = 3
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
    if real: #not error_margin. Print fitted parameters
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

#coupling = [4, 3, 2, 1]

    def plot(self):
        fig2, axs2 = plt.subplots(nrows=3, ncols=1, figsize=(4, 1*3))
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(2.7*3,3))
        fig, alone_ax = plt.subplots(nrows=1, ncols=1, figsize=(1,1.5))
        coherence = [[np.mean(coh[-1000:]) for coh in entry["phase_cohs"]] for entry in self.lines]
        coherence2 = [[coh[-1500:] for coh in entry["phase_cohs"]] for entry in self.lines]
        coherence_sems = [np.std(coh) for coh in coherence] #OBS, STD

        #longs = [np.mean(entry["Ns"])/entry["nodes"] for entry in self.lines]
        longs = [np.array(entry["Ns"])/entry["nodes"] for entry in self.lines]
        shorts = [entry["shortrange_pernode"] for entry in self.lines]

        stack = np.mean(np.array(longs),axis=1), np.array(shorts),  np.mean(np.array(coherence), axis=1), np.array(coherence_sems),
        dominance_matrix = np.vstack(stack).T
        pandas_matrix = pd.DataFrame(dominance_matrix, columns = ['g', 'k', 'r', 'r_sd'])

        from dominance_analysis import Dominance
        dominance_regression=Dominance(data=pandas_matrix,target='r_sd',objective=1)
        incr_variable_rsquare=dominance_regression.incremental_rsquare()
        print(incr_variable_rsquare)

        longs_sems = [stats.sem(long) for long in longs]
        lower_index = 0
        color_index = 0

        #predictive power
        df = pd.DataFrame()


        for index, short in enumerate(shorts):
            ax = axs[color_index]
            ax2 = axs2[color_index]
            if color_index == 0:
                cols=plt.cm.PuRd(np.linspace(0, 1, 6))
            elif color_index == 1:
                cols=plt.cm.GnBu(np.linspace(0, .7, 6))
            else:
                cols=plt.cm.BuPu(np.linspace(0, 1, 6))

            ax2.set_prop_cycle(cycler.cycler('color', cols))
            try:
                if short != shorts[index + 1]:

                    index += 1
                    x, y = np.mean(longs, axis = 1)[lower_index:index], np.mean(coherence, axis = 1)[lower_index:index]
                    x_, y_ = fit(x, y, real = True)
                    _, ey1_ = fit(x, y + coherence_sems[lower_index:index])
                    _, ey2_ = fit(x, y - coherence_sems[lower_index:index])

                    #print("Kolmogorov-Smirnov test: ", stats.ks_2samp(y, y_))

                    ax.fill_between(x_, ey2_, ey1_, color=colours[color_index],alpha=0.2, zorder=-1, lw = 0)
                    alone_ax.plot(x_, y_, c=colours[color_index], lw=1)
                    ax.plot(x_, y_, c=colours[color_index], lw=1)
                    ax.errorbar(x, y, yerr = coherence_sems[lower_index:index], fmt='o', c=colours[color_index],markersize=2, lw = 1, capsize = 1.5) #xerr=longs_sems[lower_index:index],
                    #ax.errorbar(np.mean(longs, axis = 1)[lower_index:index], np.mean(coherence, axis = 1)[lower_index:index], yerr = coherence_sems[lower_index:index], xerr=longs_sems[lower_index:index], fmt='-', c=colours[color_index],markersize=3, lw = 1, capsize = 1.5)


                    longs_ = np.array(longs)[lower_index:index,:]
                    coherence_ =  np.array(coherence)[lower_index:index,:]

                    coherence2_ =  np.array(coherence2)[lower_index:index,:,:]

                    for pl_ in range(6):
                        ax2.plot(np.arange(1500), coherence2_[3, pl_, :], lw=1)


                    df["x"] = longs_.flatten()#x
                    df["y"] = coherence_.flatten()#y

                    print(pps.score(df, "x", "y"))
                    lower_index = index
                    color_index += 1


            except Exception as e:

                x, y = np.mean(longs, axis = 1)[lower_index:], np.mean(coherence, axis = 1)[lower_index:]
                try:
                    x_, y_ = fit(x, y, real = True)
                    _, ey1_ = fit(x, y + coherence_sems[lower_index:])
                    _, ey2_ = fit(x, y - coherence_sems[lower_index:])

                except TypeError:
                    continue

                coherence2_ =  np.array(coherence2)[lower_index:,:,:]

                for pl_ in range(6):
                    ax2.plot(np.arange(1500), coherence2_[3, pl_, :], lw=1)

                ax.fill_between(x_, ey2_, ey1_, color=colours[color_index],alpha=0.2, zorder=-1, lw=0)
                ax.plot(x_, y_, c=colours[color_index], lw=1)
                alone_ax.plot(x_, y_, c=colours[color_index], lw=1)
                ax.errorbar(x, y, yerr = coherence_sems[lower_index:], fmt='o', c=colours[color_index],markersize=2, lw = 1, capsize = 1.5)

                longs_ = np.array(longs)[lower_index:,:]
                coherence_ =  np.array(coherence)[lower_index:,:]

                df["x"] = longs_.flatten()#x
                df["y"] = coherence_.flatten()#y
                print(pps.score(df, "x", "y"))
                #print("Kolmogorov-Smirnov test: ", stats.ks_2samp(y, y_))


        for index, ax in enumerate(axs):
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
            #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #plt.ticklabel_format(axis='x', style='plain')
            ax.set_ylim(0,1)
            axs2[index].set_ylim(-.025, .7)
            axs2[index].set_yticklabels([])
            axs2[index].set_xticklabels([])
            axs2[index].set_yticks([0, .7])


        alone_ax.set_xscale('log')
        alone_ax.set_ylim(0,1)

        plt.tight_layout()
        plt.show()

a = Analyser()
a.plot()
