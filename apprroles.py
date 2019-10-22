import networkx as nx
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
import argparse

from appr import appr

logging.basicConfig(format='[%(levelname)s] %(asctime)-15s %(message)s',
                    level=logging.INFO)
_LOGGER = logging.getLogger(name=os.path.basename(__file__))


def _node_descriptors(ppr_matrix, mapping):
    return {mapping[idx]: entropy(np.asarray(row)[0], base=2) for idx, row in enumerate(ppr_matrix.todense())}


def role_descriptors(dataset, alphas, epsilon, delimiter):
    # load graph
    _LOGGER.info('Loading data...')
    g = nx.read_adjlist(
        os.path.join('datasets', dataset.upper(), dataset.upper() + '_A.txt'),
        delimiter=delimiter,
        nodetype=int
    )
    labels = np.genfromtxt(
        os.path.join('datasets', dataset.upper(), dataset.upper() + '_node_labels.txt'),
        dtype=np.int
    )

    # compute role descriptors
    _LOGGER.info('Computing role descriptors...')
    descriptors = {node_id: [] for node_id in g.nodes()}
    for alpha in alphas:
        ppr_matrix, mapping = appr(g, alpha=float(alpha), epsilon=float(epsilon))
        curr_descriptors = _node_descriptors(ppr_matrix=ppr_matrix, mapping=mapping)
        for n_id, desc in curr_descriptors.items():
            descriptors[n_id].append(desc)

    # evaluate
    _LOGGER.info('Creating plot...')
    sorted_keys = sorted(list(descriptors.keys()))
    embeddings = np.array([descriptors[k] for k in sorted_keys])

    pca = PCA(n_components=1)
    X = pca.fit_transform(embeddings)
    X = minmax_scale(X)
    plt.scatter(X, y=[i for i in range(len(X))], c=labels, s=200)

    plt.ylabel('Node identifier')
    plt.yticks(np.arange(0, X.shape[0], 5))
    plt.tight_layout()
    os.makedirs(os.path.join('results', dataset.upper()), exist_ok=True)
    plt.savefig(os.path.join('results', dataset.upper(), '1d_plot.png'), facecolor='w', edgecolor='w')


def graph_classification(dataset, alphas, epsilon, k, test_fraction, delimiter):
    # load data
    _LOGGER.info('Loading data...')
    g = nx.read_adjlist(os.path.join('datasets', dataset.upper(), dataset.upper() + '_A.txt'),
                        delimiter=delimiter,
                        nodetype=int)

    g_labels = np.genfromtxt(os.path.join('datasets', dataset.upper(), dataset.upper() + '_graph_labels.txt'), dtype=int)
    g_indicators = np.genfromtxt(os.path.join('datasets', dataset.upper(), dataset.upper() + '_graph_indicator.txt'), dtype=int)

    # -1 in cases where graph ids start with 1
    g_id_subtract = 0 if np.min(g_indicators) == 0 else 1

    # calculate node descriptors per graph
    _LOGGER.info('Computing node descriptors for each network...')
    components = nx.connected_component_subgraphs(g)
    graphs_descriptors = dict()

    for comp in components:
        descriptors = {node_id: [] for node_id in comp.nodes()}
        g_id = g_indicators[list(descriptors.keys())[0]]

        for alpha in alphas:
            ppr_matrix, mapping = appr(comp, alpha=float(alpha), epsilon=float(epsilon))
            curr_descriptors = _node_descriptors(ppr_matrix=ppr_matrix, mapping=mapping)
            for n_id, desc in curr_descriptors.items():
                descriptors[n_id].append(desc)

        graphs_descriptors[g_id] = descriptors

    # collect all role descriptors
    all_role_descriptors = []
    for desc in graphs_descriptors.values():
        for d in desc.values():
            all_role_descriptors.append(d)

    # train kmeans model
    _LOGGER.info('Computing graph representations...')
    k_means = KMeans(n_clusters=k, init='k-means++', n_init=10).fit(all_role_descriptors)

    # compute representations for each graph
    graph_representations = np.zeros((g_labels.shape[0], k))
    for g_id, n_descriptors in graphs_descriptors.items():
        X = [desc for desc in n_descriptors.values()]
        labels_in_graph, counts = np.unique(k_means.predict(X), return_counts=True)
        graph_representations[g_id - g_id_subtract, labels_in_graph] = counts

    # eval
    _LOGGER.info('Doing classification...')
    ind_train, ind_test, _, _ = train_test_split(list(range(g_labels.shape[0])),
                                                 g_labels,
                                                 test_size=test_fraction,
                                                 shuffle=True,
                                                 stratify=g_labels)

    model = KNeighborsClassifier(n_neighbors=1, metric='minkowski', n_jobs=-1)
    model.fit(graph_representations[ind_train], g_labels[ind_train])

    y_test = g_labels[ind_test]
    y_pred = model.predict(graph_representations[ind_test])

    _LOGGER.info('Accuracy: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # subparser for role descriptors
    subparsers = parser.add_subparsers(help='Either use graph-classification or role-descriptors command.')
    parser_rd = subparsers.add_parser('role-descriptors', help='Compute role descriptors.')
    parser_rd.add_argument('--dataset', type=str, required=False, default='BARBELL',
                           help='Graph dataset.')
    parser_rd.add_argument('--delimiter', type=str, required=False, default=' ',
                           help='The delimiter that is used in the adjacency list file.')
    parser_rd.add_argument('--alphas', nargs='+', required=False, default=[0.1],
                           help='Alpha values for which APPR shall be computed. Can be one or many values, '
                                'default is 0.1.')
    parser_rd.add_argument('--epsilon', type=float, required=False, default=1e-4,
                           help='Approximation error bound for the APPR computation, default is 1e-4.')


    # subparser for graph classification
    parser_gc = subparsers.add_parser('graph-classification', help='Perform graph classification.')
    parser_gc.add_argument('--dataset', type=str, required=True,
                           help='Graph dataset.')
    parser_gc.add_argument('--delimiter', type=str, required=False, default=', ',
                           help='The delimiter that is used in the adjacency list file.')
    parser_gc.add_argument('--alphas', nargs='+', required=False, default=[0.1],
                           help='Alpha values for which APPR shall be computed. Can be one or many values, default is 0.1.')
    parser_gc.add_argument('--epsilon', type=float, required=False, default=1e-4,
                           help='Approximation error bound for the APPR computation, default is 1e-4.')
    parser_gc.add_argument('--test-fraction', type=float, required=False, default=0.1,
                           help='Fraction that shall be used for testing, default is 0.1.')
    parser_gc.add_argument('--k', type=int, required=False, default=5,
                           help='Number of roles, resp. dimensionality of the graph representations.')

    args = parser.parse_args()

    if 'k' in args:
        graph_classification(**vars(args))
    else:
        role_descriptors(**vars(args))
