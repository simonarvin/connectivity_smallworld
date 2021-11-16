import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import pathlib

data_name = f"sw_kuramoto_stability"
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"
data_path = f"{base_path}/{data_name}.json"

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
        fig_stability, axs = plt.subplots(nrows = 2, ncols = 4, figsize=(2 * 4, 2*2), sharey=True, sharex=True)
        fig_stability.canvas.manager.set_window_title('Stability maps')

        longrange_probabilities = np.logspace(np.log10(0.0001), np.log10(1), 9)
        coherences = np.flip(np.linspace(0, 1, 10))
        shortranges = [5, 25, 50]

        heatmap = np.zeros((len(longrange_probabilities), len(coherences)), dtype=np.float64)
        attractionmaps = [heatmap.copy(), heatmap.copy(),heatmap.copy()]
        heatmaps = [heatmap.copy(), heatmap.copy(),heatmap.copy()] #* len(shortranges)

        end_step = -1

        print(f"computing stability and attractiveness,\nend_step dt = {np.array(self.lines[0]['phase_cohs']).T.shape[0] + end_step}")

        for entry in self.lines:

            heatmap = heatmaps[shortranges.index(entry["shortrange_pernode"])]
            attractionmap = attractionmaps[shortranges.index(entry["shortrange_pernode"])]
            phase_coherence = np.array(entry["phase_cohs"]).T

            deviation =   (np.abs(np.mean(phase_coherence[end_step, :]) - entry["coherence"]))
            heatmap[np.where(longrange_probabilities == entry["longrange_probability"]), np.where(coherences == entry["coherence"])] = np.mean(deviation)

            end_regime = (np.abs(coherences - np.mean(phase_coherence[end_step, :]))).argmin()

            attractionmap[np.where(longrange_probabilities == entry["longrange_probability"]), end_regime] += -np.mean(np.abs(entry["coherence"] - np.mean(phase_coherence[end_step:,:],axis=1)))


        comb_heat = np.hstack(heatmaps)
        diff1 = heatmaps[0].T - heatmaps[1].T #H = 10 vs H = 50
        diff2 = heatmaps[1].T - heatmaps[2].T #H = 50 vs H = 100
        diff3 = heatmaps[0].T - heatmaps[2].T #H = 10 vs H = 100
        comb_diff = np.hstack((diff1, diff2, diff3))

        print("plotting stability maps..")
        print("upper row: stability maps")
        print("lower row: difference stability maps")

        for i, ax in enumerate(axs.reshape(-1)):
            if i < 3:

                ax.set_title(f"$H = {shortranges[i]*2}$", fontsize = 10)

                c= ax.pcolor(longrange_probabilities, coherences, heatmaps[i].T[:-1,:-1], vmin = np.min(comb_heat), vmax = np.max(comb_heat),cmap = "viridis")

                ax.set_yticks([0, .5, 1])

            elif i == 3:
                #colorbar
                cb = fig_stability.colorbar(c, ax = ax, ticks = [np.min(comb_heat), np.max(comb_heat)])
                continue

            elif i == 4:
                #H = 10 vs H = 50
                ax.set_title(f"$H = 10 v 50$", fontsize = 10)
                ax.pcolor(longrange_probabilities, coherences, diff1[:-1,:-1], vmin = np.min(comb_diff), vmax=np.max(comb_diff))

            elif i == 5:
                #H = 50 vs H = 100
                ax.set_title(f"$H = 50 v 100$", fontsize = 10)
                ax.pcolor(longrange_probabilities, coherences, diff2[:-1,:-1], vmin = np.min(comb_diff), vmax=np.max(comb_diff))
            elif i == 6:
                #H = 10 vs H = 100
                ax.set_title(f"$H = 10 v 100$", fontsize = 10)
                ax.pcolor(longrange_probabilities, coherences, diff3[:-1,:-1], vmin = np.min(comb_diff), vmax=np.max(comb_diff))
            else:
                break

            ax.axhline(.25, linestyle="--", color="grey")
            ax.axhline(.75, linestyle="--", color="grey")
            ax.set_xscale("log")

        plt.tight_layout()

        print("plotting attractiveness maps..")
        print("upper row: attractiveness maps")
        print("lower row: difference attractiveness maps")

        attraction_diff = attractionmaps[0].T - attractionmaps[1].T
        comb_attr = np.hstack(attractionmaps)
        attraction_fig, attraction_axs = plt.subplots(nrows = 2, ncols = 4, figsize=(2 * 4, 2*2), sharey=True, sharex=True)
        attraction_fig.canvas.manager.set_window_title('Attractiveness maps')

        for i, ax in enumerate(attraction_axs.reshape(-1)):

            if i == 3:
                attraction_fig.colorbar(c, ax = ax, ticks = [np.min(comb_heat), np.max(comb_heat)])
                continue
            elif i == 4:
                #H = 10 v 50
                ax.pcolor(longrange_probabilities, coherences, attraction_diff[:-1,:-1], cmap = "PiYG")
                ax.set_title(f"$H = 10 v 50$", fontsize = 10)

            elif i == 5:
                #H = 50 v 100
                ax.pcolor(longrange_probabilities, coherences, (attractionmaps[1].T - attractionmaps[2].T)[:-1,:-1], cmap = "PiYG")
                ax.set_title(f"$H = 50 v 100$", fontsize = 10)
            elif i == 6:
                #H = 10 v 100
                c = ax.pcolor(longrange_probabilities, coherences, (attractionmaps[0].T - attractionmaps[2].T)[:-1,:-1], cmap = "PiYG")
                ax.set_title(f"$H = 10 v 100$", fontsize = 10)
            elif i == 7:
                attraction_fig.colorbar(c, ax = ax)
                continue
            else:
                #attractiveness maps, upper row
                ax.pcolor(longrange_probabilities, coherences, attractionmaps[i].T[:-1,:-1], vmin = np.min(comb_attr), vmax = np.max(comb_attr),cmap = "viridis")
                ax.set_title(f"$H = {shortranges[i]*2}$", fontsize = 10)

            ax.set_xscale("log")

            ax.set_yticks([0, .5, 1])

            ax.axhline(.25, linestyle="--", color="grey")
            ax.axhline(.75, linestyle="--", color="grey")

        plt.tight_layout()
        print("plotting attractiveness difference curves..")
        attraction_fig, attraction_ax = plt.subplots(nrows = 1, ncols = 1, figsize=(2.6, 3.0))
        attraction_fig.canvas.manager.set_window_title('Attractiveness difference curves')

        attraction_ax.errorbar(np.mean(attraction_diff, axis = 1), coherences, xerr=stats.sem(attraction_diff, axis = 1), fmt=".-", color = "#1C9E77", label = "$H = 10 v 50$")

        attraction_diff = attractionmaps[0].T - attractionmaps[2].T
        attraction_ax.errorbar(np.mean(attraction_diff, axis = 1),coherences, xerr=stats.sem(attraction_diff, axis = 1), fmt=".--", color = "#E33237", label = "$H = 10 v 100$")

        attraction_ax.set_xticks([-3, 0, 1])

        attraction_ax.set_yticks([0, .25, .5, .75, 1])
        attraction_ax.invert_xaxis()
        attraction_ax.legend(loc="upper right")
        attraction_ax.set_xlabel("attractiveness diff")
        attraction_ax.set_ylabel("synchrony, $r$")

        attraction_ax.axhline(.25, linestyle="--", color="grey")
        attraction_ax.axhline(.75, linestyle="--", color="grey")

        attraction_ax.axvline(0, color="black")


        plt.tight_layout()
        plt.show()

a = Analyser()
a.plot()
