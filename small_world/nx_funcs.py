from networkx.utils import py_random_state
import networkx as nx

@py_random_state(4)
def arvin_watts_strogatz_graph(n, k, T, u, seed = None):
    """
    n = nodes
    k = short-range connectivity (wires/node)
    T = max long-range connectivity (wires/node)
    u = probability long-range connectivity (0-1)
    g = long-range connectivity = T * u
    """

    if k > n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    # If k == n the graph return is a complete graph
    if k == n:
        return nx.complete_graph(n)

    G = nx.empty_graph(n)

    nlist = list(G.nodes())
    fromv = nlist
    # connect the k/2 neighbors

    for j in range(1, k // 2 + 1):
        tov = fromv[j:] + fromv[0:j]  # the first j are now last
        for i in range(len(fromv)):
            G.add_edge(fromv[i], tov[i])
    # for each edge u-v, with probability p, randomly select existing
    # node w and add new edge u-w
    Gshort = G.copy()
    longrange_connections = 0

    e = list(G.nodes())

    for u_ in e:
        ws = []

        for _ in range(T):
            if seed.random() < u:
                w = seed.choice(nlist)
                # no self-loops and reject if edge u-w exists
                # is that the correct NWS model?
                while w == u_ or G.has_edge(u_, w) or w in ws:
                    w = seed.choice(nlist)
                    if G.degree(u_) >= n - 1:
                        break  # skip this rewiring
                else:

                    longrange_connections += 1
                    G.add_edge(u_, w)

                    ws.append(w)
    Glong = nx.difference(G, Gshort)
    return G, longrange_connections, Glong, Gshort
