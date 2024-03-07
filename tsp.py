import numpy as np
import time

from matplotlib import pyplot as plt

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

def plot_solution(coordinates, solution, ax, title):
    """
    Plot the TSP path on a given axes.
    """
    ordered_coords = [coordinates[i] for i in solution] + [coordinates[solution[0]]]
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys, 'o-', markersize=5, linewidth=1, label='Path')
    ax.plot(xs[0], ys[0], 'ro', markersize=8, label='Start/End')
    for i, (x, y) in enumerate(ordered_coords[:-1]):  # Exclude the last point because it's a repeat of the first
        ax.text(x, y, str(solution[i]), color="blue", fontsize=9)
    ax.set_title('Best Solution for TSP in', title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

def plot_run_statistics(distances, average_distance, std_deviation, ax, title):
    """
    Plot the performance statistics on a given axes.
    """
    runs = list(range(1, len(distances) + 1))
    ax.plot(runs, distances, 'o-', label='Distance per Run')
    ax.axhline(y=average_distance, color='r', linestyle='-', label='Average Distance')
    ax.fill_between(runs, average_distance - std_deviation, average_distance + std_deviation, color='red', alpha=0.2,
                    label='Std Deviation Range')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Distance')
    ax.set_title('Performance over Multiple Runs in', title)
    ax.legend()

def plot_both_graph(coordinates, best_solution, distances, average_distance, std_deviation, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_solution(coordinates, best_solution, axs[0], title)
    plot_run_statistics(distances, average_distance, std_deviation, axs[1], title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    file_path = 'att48.tsp'
    coordinates = parse_tsp_file(file_path)

    temp_values = np.linspace(1000,3000)  # paper Simulated annealing based symbiotic organisms search optimization algorithm for traveling salesman problem
    cooling_rates = [0.99]
    stopping_temps = [0.025]

    two_opt_attempts = 250

    num_trials = 100
    num_runs = 30
    fitness_evaluations = 10000

    start_time = time.time()
    sa.run(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, num_runs, fitness_evaluations, two_opt_attempts)
    print(f"--- Simulated Annealing time: {time.time() - start_time} seconds ---")

    population_sizes = [50]
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1]
    n_generations = 100

    total_trials = 100
    runs = 30

    start_time = time.time()
    ga.run(coordinates, total_trials, population_sizes, crossover_rates, mutation_rates, n_generations, runs)
    print(f"--- Genetic Algorithm time: {time.time() - start_time} seconds ---")
