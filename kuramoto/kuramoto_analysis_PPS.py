import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import cycler

import pandas as pd
import ppscore as pps
from dominance_analysis import Dominance
import scipy

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
data_name = f"sw_kuramoto_coupling{coupling}_suppl"
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"
data_path = f"{base_path}/{data_name}.json"

def linfunc(x, A, B):

    return A * x + B

class Analyser:
    def __init__(self):
        self.lines = []
        with open(data_path) as file:
            lines = file.readlines()
            for line in lines: # load data
                js = json.loads(line)
                self.lines.append(js)

    def plot(self):
        fig_metastability, ax_metastability = plt.subplots(nrows = 1, ncols = 1, figsize=(2, 2))

        fig_metastability.canvas.manager.set_window_title('metastability')

        coherence = [[np.mean(coh[-1000:]) for coh in entry["phase_cohs"]] for entry in self.lines]
        raw_coherence = [[coh[-1500:] for coh in entry["phase_cohs"]] for entry in self.lines]
        coherence_std = [np.std(coh) for coh in coherence] #OBS, STD

        #longs = [np.mean(entry["longrange_connections_arr"])/entry["nodes"] for entry in self.lines]
        longs = [np.array(entry["longrange_connections_arr"])/entry["nodes"] for entry in self.lines]
        shorts = [entry["shortrange_pernode"] for entry in self.lines]


        #print(np.unique(shorts))

        #check dominance of variables g, h and r onto the metastability r_std
        stack = np.mean(np.array(longs),axis=1), np.array(shorts),  np.mean(np.array(coherence), axis=1), np.array(coherence_std),
        dominance_matrix = np.vstack(stack).T
        pandas_matrix = pd.DataFrame(dominance_matrix, columns = ['g', 'h', 'r', 'r_std'])

        dominance_regression=Dominance(data = pandas_matrix,target = 'r_std',objective=1)
        incr_variable_rsquare = dominance_regression.incremental_rsquare()
        print(f"\ndominance analysis:\n{incr_variable_rsquare}\n")

        lower_index = 0

        gs = []
        pps_ = []

        for index, short in enumerate(shorts):

            try:
                if short != shorts[index + 1]:

                    index += 1

                    #compute mean curve
                    x, y = np.mean(longs, axis = 1)[lower_index:index], np.mean(coherence, axis = 1)[lower_index:index]

                    longs_scope = np.array(longs)[lower_index:index,:]
                    coherence_scope =  np.array(coherence)[lower_index:index,:]
                    raw_coherence_scope =  np.array(raw_coherence)[lower_index:index,:,:]

                    #compute the predictive score of long-range connectivity g on network synchrony r
                    df = pd.DataFrame()
                    df["g"] = longs_scope.flatten() #long-range connectivity
                    df["r"] = coherence_scope.flatten() #y
                    predictive_score = pps.score(df, "g", "r")

                    print(f"\npredictive score: {predictive_score['ppscore']}\n")


                    lower_index = index

                    gs.append(short)
                    pps_.append(predictive_score['ppscore'])

            except: #catch last plot:

                x, y = np.mean(longs, axis = 1)[lower_index:], np.mean(coherence, axis = 1)[lower_index:]

                longs_scope = np.array(longs)[lower_index:,:]
                coherence_scope =  np.array(coherence)[lower_index:,:]
                raw_coherence_scope =  np.array(raw_coherence)[lower_index:,:,:]


                #compute the predictive score of long-range connectivity g on network synchrony r
                df = pd.DataFrame()
                df["g"] = longs_scope.flatten() #long-range connectivity
                df["r"] = coherence_scope.flatten() #y
                predictive_score = pps.score(df, "g", "r")

                print(f"\npredictive score: {predictive_score['ppscore']}\n")

                gs.append(short)
                pps_.append(predictive_score['ppscore'])

        pps_ = [x for _, x in sorted(zip(gs, pps_), key=lambda pair: pair[0])]
        gs.sort()
        print(gs)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(gs, pps_)
        print("PPS linear fit: ")
        print(slope, intercept, r_value, p_value)

        ax_metastability.plot(np.linspace(0, -intercept/slope, 50), linfunc(np.linspace(0, -intercept/slope, 50), slope, intercept),"--", color="#F98E23")
        ax_metastability.scatter(gs, pps_, color="#F98E23", s=4)
        
        ax_metastability.set_ylim(0,1)
        ax_metastability.set_xscale("log")
        plt.show()


a = Analyser()
a.plot()
