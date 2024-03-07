import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt
import time

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

def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def total_distance(coords, solution):
    """Calculate the total distance of the travel path."""
    return sum(calculate_distance(coords[solution[i]], coords[solution[i - 1]]) for i in range(len(solution)))

def create_initial_population(size, n_coordinates):
    return [random.sample(range(n_coordinates), n_coordinates) for _ in range(size)]

def tournament_selection(population, scores, k=5):
    selection_ix = np.random.randint(len(population), size=k)
    selected_fitness = [scores[ix] for ix in selection_ix]
    winner_ix = selection_ix[np.argmin(selected_fitness)]
    return population[winner_ix]


def crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))

    # Copy a part of parent1 to the child
    child[start:end] = parent1[start:end]

    # Fill the remaining part from parent2
    p2_index = end
    c_index = end
    while None in child:
        if parent2[p2_index % size] not in child:
            child[c_index % size] = parent2[p2_index % size]
            c_index += 1
        p2_index += 1

    return child

def mutate(solution, coords, max_attempts=10):
    # i, j = sorted(random.sample(range(len(tour)), 2))
    # tour[i:j + 1] = reversed(tour[i:j + 1])
    best_distance = total_distance(coords, solution)
    for _ in range(max_attempts):
        start, end = sorted(random.sample(range(1, len(solution)), 2))
        new_solution = solution[:start] + solution[start:end][::-1] + solution[end:]
        new_distance = total_distance(coords, new_solution)
        if new_distance < best_distance:
            solution, best_distance = new_solution, new_distance

    return solution

def genetic_algorithm(coordinates, population_size, n_generations, crossover_rate, mutation_rate):
    # Create initial population
    population = create_initial_population(population_size, len(coordinates))
    best_distance = float('inf')
    best_tour = None

    for generation in range(n_generations):

        scores = [total_distance(coordinates, tour) for tour in population]
        for i, score in enumerate(scores):
            if score < best_distance:
                best_distance, best_tour = score, population[i]

        # Select parents
        selected = [tournament_selection(population, scores) for _ in range(population_size)]

        # Create the next generation
        children = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[i + 1]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

                # Apply mutation with inversion_mutation based on mutation_rate
            if random.random() < mutation_rate:
                child1 = mutate(child1, coordinates)
            if random.random() < mutation_rate:
                child2 = mutate(child2, coordinates)

            children.extend([child1, child2])

        population = children

    return best_tour, best_distance


def parameter_tuning(coordinates, num_trials, population_sizes, crossover_rates, mutation_rates, n_generations):
    best_params = None
    best_distance = float('inf')

    parameter_combinations = list(product(population_sizes, crossover_rates, mutation_rates))

    for trial in range(num_trials):
        params = random.choice(parameter_combinations)
        population_size, crossover_rate, mutation_rate = params
        _, distance = genetic_algorithm(coordinates, population_size, n_generations, crossover_rate, mutation_rate)
        if distance < best_distance:
            best_distance = distance
            best_params = params

    return best_params


def execute_with_tuned_parameters(coordinates, best_params, runs):
    population_size, crossover_rate, mutation_rate = best_params
    n_generations = 10000 // population_size
    best_overall_distance = float('inf')
    best_overall_tour = None

    distances = []
    for run in range(runs):
        print(f"Running run {run + 1}/{runs} with n_generations: {n_generations}", end='\r')
        tour, distance = genetic_algorithm(coordinates, population_size, n_generations, crossover_rate, mutation_rate)
        distances.append(distance)
        if distance < best_overall_distance:
            best_overall_distance = distance
            best_overall_tour = tour

    return distances, best_overall_distance, best_overall_tour, np.mean(distances), np.std(distances)

def plot_solution(coordinates, solution, ax):
    """
    Plot the TSP path on a given axes.
    """
    ordered_coords = [coordinates[i] for i in solution] + [coordinates[solution[0]]]
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys, 'o-', markersize=5, linewidth=1, label='Path')
    ax.plot(xs[0], ys[0], 'ro', markersize=8, label='Start/End')
    for i, (x, y) in enumerate(ordered_coords[:-1]):  # Exclude the last point because it's a repeat of the first
        ax.text(x, y, str(solution[i]), color="blue", fontsize=9)
    ax.set_title('Best Solution for TSP in GA')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()

def plot_run_statistics(distances, average_distance, std_deviation, ax):
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
    ax.set_title('Performance over Multiple Runs in GA')
    ax.legend()

def run(coordinates, total_trials, population_sizes, crossover_rates, mutation_rates, n_generations, runs):
    best_params = parameter_tuning(coordinates, total_trials, population_sizes, crossover_rates, mutation_rates, n_generations)
    print(f"Best Parameters: {best_params}")
    distances, best_overall_distance, best_solution, average_distance, std_deviation = execute_with_tuned_parameters(coordinates, best_params, runs)
    print(f"Best Overall Distance: {best_overall_distance}")
    print(f"Average Distance: {average_distance}")
    print(f"Standard Deviation: {std_deviation}")

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_solution(coordinates, best_solution, axs[0])
    plot_run_statistics(distances, average_distance, std_deviation, axs[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    start_time = time.time()

    file_path = 'att48.tsp'
    coordinates = parse_tsp_file(file_path)

    population_sizes = [50]
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1]
    n_generations = 100

    total_trials = 100
    runs = 30

    run(coordinates, total_trials, population_sizes, crossover_rates, mutation_rates, n_generations, runs)

    print(f"--- {time.time() - start_time} seconds ---")