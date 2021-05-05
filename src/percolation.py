#!/usr/bin/python3

import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle
from copy import deepcopy


def main():
    cm = nx.connected_caveman_graph(10, 10)
    grid = nx.grid_2d_graph(10, 10)
    line = nx.path_graph(100)
    for _ in range(10):
        cms = show_score(cm)
        print('caveman score:', cms)
        gs = show_score(grid)
        print('grid score:', gs)
        ls = show_score(line)
        print('line score:', ls)


def average(xs):
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2]


def show_score(G: nx.Graph):
    phi = 1 - .75
    scores = []
    for _ in range(1001):
        G1 = deepcopy(G)
        edges = list(G.edges)
        shuffle(edges)
        G1.remove_edges_from(edges[:int(len(edges)*phi)])
        components = list(nx.connected_components(G1))
        # score = len(components)
        score = len(max(components, key=lambda c: len(c)))
        scores.append(score)
    # print(f'average = {average(scores)} median = {median(scores)}')
    return average(scores), median(scores)


if __name__ == '__main__':
    main()
