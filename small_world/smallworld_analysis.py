import json
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

from scipy import stats
import pandas as pd
from dominance_analysis import Dominance

import pathlib

base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"

# set universal plot parameters
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams.update({'errorbar.capsize': 1.5})
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
colours = ["#E33237","#1B9E77","#2E3192"]
grey = "#E6E6E6"

class Analyser:
    def __init__(self, nodes = 1000):
        self.nodes = nodes

        self.entries = self.load_entries()
        self.data = self.data_split()
        self.split_indices = self.get_split_indices()

    def split_arr(self, l, n:int):
        n = int(n)
        return np.array([l[i:i + n] for i in range(0, len(l), n)])


    def load_entries(self):
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

    def data_split(self):
        entries = self.entries

        data = {
        "shortrange_pernode" : np.array([d["shortrange_pernode"] for d in entries]),
        "longrange_probability" : np.array([d["longrange_probability"] for d in entries]),
        "omegas": np.array([d["omega_list"] for d in entries]),
        "max_longrange" : np.array([d["max_longrange"] for d in entries]),
        "longrange_connections_arr" : np.array([d["longrange_connections_arr"] for d in entries])/self.nodes
        }

        print("data succesfully parsed")

        return data

    def get_split_indices(self):

        longrange_probability = np.logspace(np.log10(0.0001), np.log10(1), 12)

        shortrange_pernode = [5, 10, 30, 50, 70, 90, 100]

        max_longrange = [10, 50, 200]
        samples = 100

        split_indices = {
        "short_len" : len(shortrange_pernode),
        "short_split" : len(longrange_probability),
        "p_split" : len(shortrange_pernode) * len(longrange_probability),
        "samples" : samples,
        "beta" : longrange_probability,
        "p_len" : len(max_longrange)
        }

        print("split indices computed")

        return split_indices

    def dominance(self, plot = True):
        split_indices = self.split_indices
        data = self.data

        entries = data["longrange_connections_arr"].shape[1]
        k_ = np.zeros(entries)
        g_ = k_.copy()

        for entry in range(entries):

            df = {
            "omega": data["omegas"][:, entry],
            "k" : data["shortrange_pernode"],
            "g" : data["longrange_connections_arr"][:, entry]
            }

            # crop to p_pernode = 10
            df = {
            "omega": data["omegas"][:split_indices["p_split"], entry],
            "k" : data["shortrange_pernode"][:split_indices["p_split"]],
            "g" : data["longrange_connections_arr"][:split_indices["p_split"], entry]
            }

            df = pd.DataFrame(df)

            dominance_regression = Dominance(data=df, target='omega', objective=1)
            dom = dominance_regression.incremental_rsquare()

            k_[entry] = dom["k"]
            g_[entry] = dom["g"]

        print("Dominance analysis:")
        print(f"k - mean: {np.mean(k_)} SEM: {stats.sem(k_)}")
        print(f"g - mean: {np.mean(g_)} SEM: {stats.sem(g_)}")
        print(stats.ttest_ind(k_, g_))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize = (.8, 1.2))
            fig.canvas.manager.set_window_title('Dominance analysis')

            ax.bar(["k", "g"], [np.mean(k_), np.mean(g_)], yerr = [stats.sem(k_), stats.sem(g_)], width=.5, color = ["#666666", "#E33237"], linewidth = 1, edgecolor="black")
            ax.tick_params(which="both", axis ="both", width=1)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .5, 1])
            plt.show()

    def plot_critical_space(self, *axs, threshold = .5):
        for ax in axs:
            ax.axhline(threshold, 0, 1, linestyle = "--", color = grey,dashes=(4, 4))
            ax.axhline(-threshold, 0, 1, linestyle = "--", color = grey,dashes=(4, 4))

    def omega_over_k(self, plot = True):
        split_indices = self.split_indices
        data = self.data

        omega_extract = data["omegas"][:split_indices["p_split"]] #extract first long-range density = 10
        omega_split = self.split_arr(omega_extract, split_indices["short_split"]) #then split according to short-range density
        omega_split = np.swapaxes(omega_split, 0, 1)

        longrange_extract = data["longrange_connections_arr"][:split_indices["p_split"]] #do the same for long-range connectivity
        longrange_split = self.split_arr(longrange_extract, split_indices["short_split"]) #then split according to short-range density
        longrange_split = np.swapaxes(longrange_split, 0, 1)

        shortrange_extract = data["shortrange_pernode"][:split_indices["p_split"]] #do the same for long-range connectivity
        shortrange_split = self.split_arr(shortrange_extract, split_indices["short_split"]) #then split according to short-range density
        shortrange_split = np.swapaxes(shortrange_split, 0, 1)

        curves = [0, 4, 8, 11] #long-range curves
        #curves = [1, 3, 6]

        if plot:
            fig, ax = plt.subplots(1, 1, figsize = (2.5, 3))
            fig.canvas.manager.set_window_title('short-range, small-worldness')

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

    def omega_over_g(self, plot = True):
        split_indices = self.split_indices
        data = self.data

        omega_extract = data["omegas"][:split_indices["p_split"]] #extract first long-range density = 10
        omega_split = self.split_arr(omega_extract, split_indices["short_split"]) #then split according to short-range density

        longrange_extract = data["longrange_connections_arr"][:split_indices["p_split"]] #do the same for long-range connectivity

        longrange_split = self.split_arr(longrange_extract, split_indices["short_split"]) #then split according to short-range density
        curves = [1, 3, 6] #short-range curves

        if plot:
            fig, ax = plt.subplots(1, 1, figsize = (2.5, 3))
            fig.canvas.manager.set_window_title('long-range, small-worldness')

            fig_small, ax_small = plt.subplots(1, 1, figsize = (2.5/3, .8))
            fig_small.canvas.manager.set_window_title('inset')

            for i, curve in enumerate(curves):
                print(f"short-range density: {data['shortrange_pernode'][curve * split_indices['short_split']]}")

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

    def mobility(self, plot_ratio = False):
        split_indices = self.split_indices
        data = self.data

        omega_split_p, longrange_split, curves_p = self.omega_over_g(plot = False)
        omega_split_k, shortrange_split, curves_k = self.omega_over_k(plot = False)

        fig, axs = plt.subplots(1, 2, figsize = (2.5 * 2, 2))
        fig.canvas.manager.set_window_title('mobility')
        beta_shape = np.resize(split_indices["beta"], omega_split_p[0].shape)

        for i, curve in enumerate(curves_p):

            x, y = longrange_split[curve], omega_split_p[curve]
            a, b = np.diff(y, axis = 0), np.diff(x, axis = 0)
            y = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
            axs[0].errorbar(np.mean(x, axis = 1), np.mean(y, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(y, axis = 1), c = colours[i],ls = "None")
            axs[0].plot(np.mean(x, axis = 1), np.mean(y, axis = 1), ".-", color = colours[i])

            if i == 0: #if k = 100
                x_k100, y_k100 = x, y

        if plot_ratio: #plot ratio yk100/yk10
            fig_small, ax_small = plt.subplots(1, 1, figsize = (2.5/3, .8))
            fig_small.canvas.manager.set_window_title('ratio, inset')
            yr_= np.divide(y_k100, y, out=np.zeros_like(y_), where=y!=0) #yk100/yk10
            ax_small.plot(np.mean(x, axis = 1), np.mean(yr_, axis = 1),".-", c=colours[0])
            ax_small.errorbar(np.mean(x, axis = 1), np.mean(yr_, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(yr_, axis = 1), c = colours[0],ls = "None")
            ax_small.axhline(1, 0, 1, color = "black")
            ax_small.fill_between(np.mean(x, axis = 1), 1, np.mean(yr_, axis=1), facecolor=colours[0], alpha=.2)
            ax_small.set_xscale("log")

            ax_small.set_xticks([0.01,10])
            ax_small.tick_params(which="both", axis ="both", width=1)
            ax_small.set_ylim(0, 7)

        y_diff = y - y_k100 #k10 - k100

        axs[1].plot(np.mean(x, axis=1), np.mean(y_diff, axis=1), ".-", c = colours[0])
        axs[1].errorbar(np.mean(x, axis = 1), np.mean(y_diff, axis = 1), xerr = stats.sem(x, axis = 1), yerr = stats.sem(y_diff, axis = 1), c = colours[0],ls = "None")


        for curve in curves_k: #plot short-range mobility curves
            x, y = np.resize(shortrange_split[curve],omega_split_k[curve].shape), omega_split_k[curve]
            a, b = np.diff(y, axis = 0), np.diff(x, axis = 0)
            y = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            x = (np.array(x)[:-1] + np.array(x)[1:]) / 2
            axs[0].plot(x, np.mean(y, axis = 1), ".-", color = "#666666")

        # adjust plot parameters, visual
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


a = Analyser()

a.mobility()

a.omega_over_k()
a.omega_over_g()

a.dominance(plot = False)
