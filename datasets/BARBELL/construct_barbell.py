import networkx as nx
import numpy as np

###
# PARAMS
###
clique_members = 10
chain_length = 11

# graph construction
first_clique = nx.complete_graph(clique_members)
second_clique = nx.complete_graph(clique_members)
mapping = {i: clique_members+chain_length+i for i in range(second_clique.number_of_nodes())}
second_clique = nx.relabel_nodes(second_clique, mapping=mapping)
G = nx.union(first_clique, second_clique)
for n in range(clique_members, clique_members+chain_length+1):
    G.add_edge(n-1, n)

# labels
labels = np.zeros(G.number_of_nodes(), dtype=int)
label = 0
is_odd = chain_length % 2 > 0
mid = clique_members + int(chain_length/2.)
for n in range(clique_members-1, clique_members+chain_length+1):
    if n < mid:
        label += 1
    elif n == mid and is_odd:
        label += 1
    elif n == mid and not is_odd:
        label += 0
    else:
        label -= 1

    labels[n] = label

assert label == 1

# write edge list and node labels
nx.write_edgelist(G, 'BARBELL_A.txt', delimiter=' ', data=False)
np.savetxt('BARBELL_node_labels.txt', labels, delimiter=' ', fmt='%i')
