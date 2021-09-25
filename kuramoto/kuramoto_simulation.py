import json
import numpy as np

import sys
import time
import pathlib

sys.path.insert(0,'./small_world')
import nx_funcs
import networkx as nx
from kuramoto import Kuramoto

def kuramoto(G, coupling):

    print("instantiating kuramoto")
    graph = nx.to_numpy_array(G)

    # Instantiate model with parameters
    model = Kuramoto(coupling=coupling, dt=kuramoto_dt, T=kuramoto_time, n_nodes=len(graph))
    print("success")

    # Run simulation - output is time series for all nodes (node vs time)
    print("running kuramoto simulation")
    act_mat = model.run(adj_mat=graph)
    print("success")

    print("computing kuramoto phase coherence")
    phase_coh =  [Kuramoto.phase_coherence(vec) for vec in act_mat.T]
    print("success")

    return phase_coh

try:
    start_at_trial = int(sys.argv[2])
except:
    start_at_trial = 0

try:
    start_at_coupling = int(sys.argv[1])
    c_set = True
except:
    start_at_coupling = 0
    c_set = False


#instantiate graph parameters
nodes = 1000
longrange_probabilities = np.logspace(np.log10(0.0001), np.log10(1), 10)
shortrange_pernode = [5, 25, 50]
max_longrange = [10]

#instantiate kuramoto parameters
kuramoto_dt = .1
kuramoto_time = 200 # * kuramoto_dt

couplings = [4, 3, 2, 1, 2.5]
couplings = couplings[start_at_coupling:]

#instantiate experimental parameters
total_trials_samples = 15
sample_range = range(total_trials_samples)
total_trials = len(shortrange_pernode) * len(longrange_probabilities) * len(max_longrange)
timestr = time.strftime("%Y%m%d-%H%M%S")
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"

print("\nStarting simulation tool")

for coupling in couplings:

    file_name = f"{base_path}/sw_kuramoto_coupling{coupling}_{timestr}.json"
    print(f"current coupling: {coupling}")
    print(f"save dir: {file_name}\n")

    n = 1

    for T in max_longrange:
        for k in shortrange_pernode:
            for u in longrange_probabilities:

                if start_at_trial > n:
                    n += 1
                    continue

                try:
                    print(f"generating graph:\nmax_longrange: {T}, shortrange_pernode: {k}, long-range probability: {u}\ncoupling: {coupling}")
                    print(f"Graph {n}/{total_trials}")

                    longrange_connections_arr = np.zeros(total_trials_samples)
                    phase_cohs = []

                    for sample in sample_range: #run samples
                        G, longrange_connections, _, _ = nx_funcs.arvin_watts_strogatz_graph(n = nodes, k = k, T = T, u = u)
                        phase_cohs.append(kuramoto(G, coupling))

                        longrange_connections_arr[sample] = longrange_connections
                        print(f"data appended, {sample}/{total_trials_samples}")

                    #pack dictionary of data for logging
                    dict = {
                    "kuramoto_time": kuramoto_time,
                    "kuramoto_dt" : kuramoto_dt,
                    "samples": int(total_trials_samples),
                    "nodes": int(nodes),
                    "longrange_probability": float(u),
                    "max_longrange": int(T),
                    "shortrange_pernode": float(k),
                    "longrange_connections_arr": longrange_connections_arr.tolist(),
                    "phase_cohs": phase_cohs
                    }

                    #save logged data
                    with open(file_name, 'a+') as json_file:
                        json.dump(dict, json_file)
                        json_file.write("\n")

                    print(f"Data saved: {file_name}\n\n")

                    n+=1
                except Exception as e:
                    print(f"Graph skipped. Error:\n{e}")
    if c_set:
        break

print("Computation done; File saved.")
