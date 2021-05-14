#!/usr/bin/python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from random import shuffle
from typing import Dict, Tuple, Optional

Layout = Dict[int, Tuple[float, float]]


def main():
    num_trials = 5000
    phi = .75
    name = 'Erdos-Renyi 500-.01'
    G = nx.fast_gnp_random_graph(500, .01)
    results = [analyze_percolation(percolate(G, phi)) for _ in tqdm(range(num_trials))]
    num_components, size_largest_components = zip(*results)
    plt.title(f'{name} Number of Components')
    plt.hist(num_components, bins=None)
    plt.figure()
    plt.title(f'{name} Size of Largest Component')
    plt.hist(size_largest_components, bins=None)
    plt.show()


def analyze_percolation(G: nx.Graph):
    components = list(nx.connected_components(G))
    largest_component = max(components, key=lambda comp: len(comp))
    return len(components), len(largest_component)


def average(xs):
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2]


def simulate(M: np.ndarray, beta: float, tau: int, num_infectious: int):
    pass


def percolate(G: nx.Graph, phi: float) -> nx.Graph:
    """
    :param M: Adjacency matrix
    :param phi: proportion of edges to keep
    """
    G = nx.Graph(G)
    edges = list(G.edges)
    shuffle(edges)
    G.remove_edges_from(edges[:int((1-phi)*len(edges))])
    return G


def read_file(fileName) -> Tuple[np.ndarray, Optional[Layout]]:
    with open(fileName, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype='uint8')

        line = f.readline()[:-1]
        i = 1
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
            i += 1

        rest_of_lines = tuple(map(lambda s: s.split(),
                              filter(lambda s: len(s) > 1, f.readlines())))
        layout = {int(line[0]): (float(line[1]), float(line[2]))
                  for line in rest_of_lines} if len(rest_of_lines) > 0 else None
    return matrix, layout


if __name__ == '__main__':
    main()
