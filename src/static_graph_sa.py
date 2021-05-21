from typing import Callable
from tqdm import tqdm
import networkx as nx
import numpy as np
from collections import defaultdict


def sparse_graph_objective(encoding) -> int:
    G = nx.Graph(encoding_to_adj_matrix(encoding))
    energy = len(encoding) if not nx.is_connected(G) else np.sum(encoding)
    return energy


def make_neighbor_explorer() -> Callable:
    encoding_to_tried_edges = defaultdict(lambda: set())

    def network_neighbor(encoding):
        if len(encoding_to_tried_edges[encoding]) == len(encoding):
            return encoding

        edge = np.random.randint(len(encoding))
        while edge in encoding_to_tried_edges[encoding]:
            edge = np.random.randint(len(encoding))
        encoding_to_tried_edges[encoding].add(edge)
        neighbor = np.copy(encoding)
        neighbor[edge] = 1 - neighbor[edge]
        return neighbor

    return network_neighbor
