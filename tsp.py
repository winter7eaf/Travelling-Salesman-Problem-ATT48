import numpy as np
import time

from matplotlib import pyplot as plt

from scipy.stats import wilcoxon

import simulated_annealing as sa
import genetic_algorithm as ga

def parse_tsp_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        data = file.readlines()
        start_parsing = False
        for line in data:
            if "NODE_COORD_SECTION" in line:
                start_parsing = True
                continue
            elif "EOF" in line:
                break
            if start_parsing:
                _, x, y = line.split()
                coordinates.append((float(x), float(y)))
    return coordinates

if __name__ == '__main__':

    file_path = 'att48.tsp'
    coordinates = parse_tsp_file(file_path)

    temp_values = np.linspace(1000,3000)
    cooling_rates = [0.99]
    stopping_temps = [0.005, 0.025]

    reverse_attempts = np.linspace(150,200)

    num_trials = 100
    num_runs = 30
    fitness_evaluations = 10000

    start_time = time.time()
    sa_distances = sa.run(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, num_runs, fitness_evaluations, reverse_attempts)
    print(f'Distances: {sa_distances}')
    print(f"--- Simulated Annealing time: {time.time() - start_time} seconds ---")

    population_sizes = [50]
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1]
    n_generations = 100

    total_trials = 100
    runs = 30

    start_time = time.time()
    ga_distances = ga.run(coordinates, total_trials, population_sizes, crossover_rates, mutation_rates, n_generations, runs)
    print(f'Distances: {ga_distances}')
    print(f"--- Genetic Algorithm time: {time.time() - start_time} seconds ---")

    statistic, p_value = wilcoxon(sa_distances, ga_distances)
    print(f"Statistic: {statistic}, p-value: {p_value}")
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between SA and GA.")
    else:
        print("There is no significant difference between SA and GA.")