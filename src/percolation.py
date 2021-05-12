#!/usr/bin/python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from random import shuffle
from copy import deepcopy


def main():
    num_sims = 1000
    M = nx.connected_caveman_graph(10, 10)
    results = [simulate(M, beta, tau, num_infectious)
               for _ in tqdm(range(num_sims))]


def average(xs):
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs)//2]


def simulate(M: np.ndarray, beta: float, tau: int, num_infectious: int):
    pass


if __name__ == '__main__':
    main()
