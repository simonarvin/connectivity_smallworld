import networkx as nx
import numpy as np
import json
import sys
import time
import nx_funcs

import pathlib
base_path = f"{pathlib.Path(__file__).parent.absolute()}/data"

def C_L(G, type = 0):
    """
    computes the clustering and transitivity for lattice, random, and small-world graphs
    """
    if type == 0: #small-world/real graph
        C = nx.transitivity(G)
        L = nx.average_shortest_path_length(G)
        return C, L
    elif type == 1: #ideal lattice
        C = nx.transitivity(G)
        return C
    else: #ideal random graph
        L = nx.average_shortest_path_length(G)
        return L


longrange_probabilities = np.logspace(np.log10(0.0001), np.log10(1), 12)
longrange_probabilities = np.append(longrange_probabilities, [1.5, 2])

shortrange_pernode = [5, 10, 30, 50, 70, 90, 100]#np.arange(lower, upper, 5)

max_longrange = [10, 50, 200]

start_at_trial = 0

nodes = 500

total_trials_samples = 5
sample_range = range(total_trials_samples)
total_trials = len(shortrange_pernode) * len(longrange_probabilities) * len(max_longrange)

print("\n starting simulation tool")

timestr = time.strftime("%Y%m%d-%H%M%S")

file_name = f"{base_path}/smallworld_data_{timestr}.json"
print(f"Saving to {file_name}\n")

n = 1

for T in max_longrange:
    for k in shortrange_pernode:
        for u in longrange_probabilities:
            if start_at_trial > n:
                n += 1
                continue
            try:

                print(f"generating graph:\nmax_longrange: {T}, shortrange_pernode: {k}, long-range probability: {u}")
                print(f"Graph {n}/{total_trials}")

                C_latt_list = np.zeros(total_trials_samples)
                L_rand_list = C_latt_list.copy()
                C_list = C_latt_list.copy()
                L_list = C_latt_list.copy()
                longrange_connections_arr = C_latt_list.copy()

                for sample in sample_range:

                    G_rand, _, _, _ = nx_funcs.arvin_watts_strogatz_graph(n = nodes, k = k, T = T, u = 1)
                    print("Random graph generated     ", end='\r')
                    G_latt, _, _, _ = nx_funcs.arvin_watts_strogatz_graph(n = nodes, k = k, T = T, u = 0)
                    print("Lattice generated          ", end='\r')

                    L_rand_list[sample] = C_L(G_rand, 2)
                    C_latt_list[sample] = C_L(G_latt, 1)
                    print("Ideal rand/latt computed")

                    G, longrange_connections, _, _ = nx_funcs.arvin_watts_strogatz_graph(n = nodes, k = k, T = T, u = u)
                    print("Real graph generated      ", end='\r')
                    C_list[sample], L_list[sample]= C_L(G)

                    longrange_connections_arr[sample] = longrange_connections
                    print(f"Sample appended, {sample}/{total_trials_samples}")


                L_rand=np.mean(L_rand_list)
                L =np.mean(L_list)
                C = np.mean(C_list)
                C_latt = np.mean(C_latt_list)


                omega = L_rand/L - C/C_latt
                omega_l = np.array(L_rand_list)/np.array(L_list) - np.array(C_list)/np.array(C_latt_list)
                print("Omega, ", omega)

                dict = {
                "samples": int(total_trials_samples),
                "nodes": int(nodes),
                "longrange_probability": float(u),
                "shortrange_pernode": float(k),
                "omega": float(omega),
                "L_rand": float(L_rand),
                "C_latt": float(C_latt),
                "L": float(L),
                "C": float(C),
                "omega_list": omega_l.tolist(),
                "longrange_connections_arr": longrange_connections_arr.tolist(),
                "max_longrange": int(T)
                }

                # dict = {
                # "randint": int(total_trials_samples),
                # "nodes": int(nodes),
                # "beta": float(u),
                # "shortrange_pernode": float(shortpernode),
                # "omega": float(omega),
                # "L_rand": float(L_rand),
                # "C_latt": float(C_latt),
                # "L": float(L),
                # "C": float(C),
                # "omega_list": omega_l.tolist(),
                # "longrange_connections_arr": longrange_connections_arr.tolist(),
                # "p_pernode": int(ppernode)
                # }

                with open(file_name, 'a+') as json_file:
                    json.dump(dict, json_file)
                    json_file.write("\n")

                print(f"Data saved: {file_name}\n\n")

                n+=1
            except Exception as e:
                print(f"Graph skipped. Error:\n{e}")

print("Computation done; File saved.")
