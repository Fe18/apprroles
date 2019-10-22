import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
import time
from pairs_heap import Heap


def appr(g, alpha, epsilon, node_id=None):
    P = lil_matrix((g.number_of_nodes(), g.number_of_nodes()), dtype=np.float)
    mapping = list(g.nodes())
    for idx, n in enumerate(mapping):
        if node_id is not None and node_id != n:
            continue

        h = Heap()
        h.push(1., n)

        while True:
            residual, v = get_pushing_node(h, epsilon)
            if v is not None:
                P, h = push(idx, v, residual, mapping, g, h, P, alpha)
            else:
                break
    if node_id is None:
        return P.tocoo(), mapping_to_dict(mapping)
    else:
        return P[mapping.index(node_id)].tocoo(), mapping_to_dict(mapping)


def push(current_node, pushing_vertex, curr_residual, mapping, g, h, P, alpha):
    P[current_node, mapping.index(pushing_vertex)] += (2*alpha / (1 + alpha)) * curr_residual
    for ngh in g.neighbors(pushing_vertex):
        res_v_pair = h.get_entry(ngh)
        if res_v_pair is None:
            h.push(((1-alpha) / (1+alpha)) * curr_residual / g.degree(pushing_vertex) / g.degree(ngh), ngh)
        else:
            h.update(res_v_pair[0] + ((1-alpha) / (1+alpha)) * curr_residual / g.degree(pushing_vertex) / g.degree(ngh), ngh)
    h.remove(pushing_vertex)
    return P, h


def get_pushing_node(h, epsilon):
    for residual, v in h:
        if epsilon <= residual:  # / graph.degree(v): division by degree is done before adding the residual value to the heap entries
            return residual, v
        else:
            return None, None
    return None, None


def mapping_to_dict(mapping):
    return {idx: node_id for idx, node_id in enumerate(mapping)}


if __name__ == '__main__':
    from scipy.stats import entropy
    from sklearn.preprocessing import minmax_scale
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    g = nx.read_edgelist('datasets/barbell/edges.txt', delimiter=' ', nodetype=int)
    start = time.time()
    ppr_matrix, mapping = appr(g, alpha=.05, epsilon=1e-6)

    descriptors = {mapping[idx]: entropy(np.asarray(row)[0]) for idx, row in enumerate(ppr_matrix.todense())}

    sorted_keys = sorted(list(descriptors.keys()))
    embeddings = np.array([descriptors[k] for k in sorted_keys])
    labels = np.genfromtxt('datasets/barbell/labels.txt', dtype=np.int)

    pca = PCA(n_components=1)
    X = pca.fit_transform(embeddings.reshape(-1, 1))
    X = minmax_scale(X)
    plt.scatter(X, y=[i for i in range(len(X))], c=labels, s=200)

    plt.ylabel('Node identifier')
    plt.yticks(np.arange(0, X.shape[0], 5))
    plt.show()
