import numpy as np
import random
import math
from itertools import product
import matplotlib.pyplot as plt


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
    return sum(calculate_distance(coords[solution[i]], coords[solution[i - 1]]) for i in range(len(solution)))


def generate_initial_solution(coords):
    solution = list(range(len(coords)))
    random.shuffle(solution)
    return solution


def generate_neighbor(solution):
    a, b = random.sample(range(len(solution)), 2)
    solution[a], solution[b] = solution[b], solution[a]
    return solution


def reverse(coords, solution, max_attempts):
    best_distance = total_distance(coords, solution)
    for _ in range(max_attempts):
        start, end = sorted(random.sample(range(1, len(solution)), 2))
        new_solution = solution[:start] + solution[start:end][::-1] + solution[end:]
        new_distance = total_distance(coords, new_solution)
        if new_distance < best_distance:
            solution, best_distance = new_solution, new_distance
    return solution, best_distance


def simulated_annealing(coords, max_iterations, temp, cooling_rate, stopping_temp, reverse_attempts):
    current_solution = generate_initial_solution(coords)
    current_distance = total_distance(coords, current_solution)
    best_solution = list(current_solution)
    best_distance = current_distance

    iteration = 0
    while temp > stopping_temp and iteration < max_iterations:
        new_solution = generate_neighbor(list(current_solution))
        new_distance = total_distance(coords, new_solution)

        if new_distance < best_distance:
            best_solution, best_distance = new_solution, new_distance

        if new_distance < current_distance or math.exp((current_distance - new_distance) / temp) > random.random():
            current_solution, current_distance = new_solution, new_distance

            current_solution, current_distance = reverse(coords, current_solution, reverse_attempts)
            current_distance = total_distance(coords, current_solution)

            if current_distance < best_distance:
                best_solution, best_distance = current_solution, current_distance

        temp *= cooling_rate
        iteration += 1

    return best_solution, best_distance


def conduct_trials(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, reverse_attempts):
    trial_results = []

    parameter_combinations = list(product(temp_values, cooling_rates, stopping_temps, reverse_attempts))

    for trial in range(num_trials):
        params = random.choice(parameter_combinations)
        temp, cooling_rate, stopping_temp, reverse_attempt = params
        reverse_attempt = int(reverse_attempt)
        solution, sa_distance = simulated_annealing(coordinates, 1, temp, cooling_rate, stopping_temp, reverse_attempt)
        trial_results.append((temp, cooling_rate, stopping_temp, reverse_attempt, sa_distance))

    trial_results_sorted = sorted(trial_results, key=lambda x: x[3])
    best_params = trial_results_sorted[0]

    return best_params


def perform_multiple_runs(coords, num_runs, best_params, fitness_evaluations):
    temp, cooling_rate, stopping_temp, two_opt_attempts, _ = best_params
    distances = []
    best_overall_distance = float('inf')
    best_overall_solution = None

    for run in range(num_runs):
        solution, distance = simulated_annealing(coords, fitness_evaluations, temp, cooling_rate, stopping_temp,
                                                 two_opt_attempts)
        # print(f'Distance for run {run + 1}: {distance}')
        distances.append(distance)
        if distance < best_overall_distance:
            best_overall_distance = distance
            best_overall_solution = solution

    average_distance = np.mean(distances)
    std_deviation = np.std(distances)
    return distances, best_overall_distance, best_overall_solution, average_distance, std_deviation

def plot_improvement_over_time(trial_best_distances):
    plt.figure(figsize=(10, 5))
    plt.plot(trial_best_distances, 'o-', color='blue')
    plt.title('Improvement over Trials')
    plt.xlabel('Trial')
    plt.ylabel('Best Distance Found')
    plt.grid(True)
    plt.show()

def plot_solution(coordinates, solution, ax):
    ordered_coords = [coordinates[i] for i in solution] + [coordinates[solution[0]]]
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys, 'o-', markersize=5, linewidth=1, label='Path')
    ax.plot(xs[0], ys[0], 'ro', markersize=8, label='Start/End')
    for i, (x, y) in enumerate(ordered_coords[:-1]):  # Exclude the last point because it's a repeat of the first
        ax.text(x, y, str(solution[i]), color="blue", fontsize=9)
    ax.set_title('Best Solution for TSP in SA')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()


def plot_run_statistics(distances, average_distance, std_deviation, ax):
    runs = list(range(1, len(distances) + 1))
    ax.plot(runs, distances, 'o-', label='Distance per Run')
    ax.axhline(y=average_distance, color='r', linestyle='-', label='Average Distance')
    ax.fill_between(runs, average_distance - std_deviation, average_distance + std_deviation, color='red', alpha=0.2,
                    label='Std Deviation Range')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Distance')
    ax.set_title('Performance over Multiple Runs in SA')
    ax.legend()


def run(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, num_runs, max_iterations,
        two_opt_attempts):
    best_params = conduct_trials(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, two_opt_attempts)
    print("Using best parameters from trials:", best_params)

    distances, best_overall_distance, best_solution, average_distance, std_deviation = perform_multiple_runs(
        coordinates, num_runs, best_params, max_iterations)
    print(f"Best Overall Distance (Simulated Annealing): {best_overall_distance}")
    print(f"Average Distance (Simulated Annealing): {average_distance}")
    print(f"Standard Deviation (Simulated Annealing): {std_deviation}")

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_solution(coordinates, best_solution, axs[0])
    plot_run_statistics(distances, average_distance, std_deviation, axs[1])
    plt.tight_layout()
    plt.show()

    return distances
