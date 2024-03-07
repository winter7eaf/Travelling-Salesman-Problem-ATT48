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
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def total_distance(coords, solution):
    """Calculate the total distance of the travel path."""
    return sum(calculate_distance(coords[solution[i]], coords[solution[i - 1]]) for i in range(len(solution)))

def generate_initial_solution(coords):
    """Generate an initial random solution."""
    solution = list(range(len(coords)))
    random.shuffle(solution)
    return solution

def generate_neighbor(solution):
    """Generate a neighbor solution by swapping two cities."""
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

def simulated_annealing(coords, max_iterations, temp, cooling_rate, stopping_temp, two_opt_attempts):
    """Solve the TSP using Simulated Annealing."""
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

            current_solution, current_distance = reverse(coords, current_solution, two_opt_attempts)
            # Since the 2-opt might have improved the solution, we need to re-calculate the distance
            current_distance = total_distance(coords, current_solution)

            # Update the best solution found so far if the new solution is better
            if current_distance < best_distance:
                best_solution, best_distance = current_solution, current_distance

        temp *= cooling_rate
        iteration += 1

    return best_solution, best_distance

def conduct_trials(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, two_opt_attempts):
    trial_results = []

    parameter_combinations = list(product(temp_values, cooling_rates, stopping_temps))

    if len(parameter_combinations) > num_trials:
        parameter_combinations = random.sample(parameter_combinations, num_trials)

    for params in parameter_combinations:
        temp, cooling_rate, stopping_temp = params
        solution, sa_distance = simulated_annealing(coordinates, 1, temp, cooling_rate, stopping_temp, two_opt_attempts)
        trial_results.append((temp, cooling_rate, stopping_temp, sa_distance))


    trial_results_sorted = sorted(trial_results, key=lambda x: x[3])
    best_params = trial_results_sorted[0]

    return best_params

def perform_multiple_runs(coords, num_runs, best_params, fitness_evaluations, two_opt_attempts):
    temp, cooling_rate, stopping_temp,_ = best_params
    distances = []
    best_overall_distance = float('inf')
    best_overall_solution = None

    for run in range(num_runs):
        solution, distance = simulated_annealing(coords, fitness_evaluations, temp, cooling_rate, stopping_temp, two_opt_attempts)
        distances.append(distance)
        if distance < best_overall_distance:
            best_overall_distance = distance
            best_overall_solution = solution

    average_distance = np.mean(distances)
    std_deviation = np.std(distances)
    return distances, best_overall_distance, best_overall_solution, average_distance, std_deviation

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
    ax.set_title('Best Solution for TSP in SA')
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
    ax.set_title('Performance over Multiple Runs in SA')
    ax.legend()

def run(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, num_runs, max_iterations, two_opt_attempts):
    best_params = conduct_trials(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, two_opt_attempts)
    print("Using best parameters from trials:", best_params)

    distances, best_overall_distance, best_solution, average_distance, std_deviation = perform_multiple_runs(coordinates, num_runs, best_params, max_iterations, two_opt_attempts)
    print(f"Best Overall Distance (Simulated Annealing): {best_overall_distance}")
    print(f"Average Distance (Simulated Annealing): {average_distance}")
    print(f"Standard Deviation (Simulated Annealing): {std_deviation}")

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_solution(coordinates, best_solution, axs[0])
    plot_run_statistics(distances, average_distance, std_deviation, axs[1])
    plt.tight_layout()
    plt.show()

# Main execution code
if __name__ == "__main__":

    file_path = 'att48.tsp'
    coordinates = parse_tsp_file(file_path)

    temp_values = np.linspace(1000, 3000) #paper Simulated annealing based symbiotic organisms search optimization algorithm for traveling salesman problem
    cooling_rates = [0.99]
    stopping_temps = [0.025]

    two_opt_attempts = 250
    num_trials = 100
    num_runs = 30
    fitness_evaluations = 10000

    run(coordinates, num_trials, temp_values, cooling_rates, stopping_temps, num_runs, fitness_evaluations, two_opt_attempts)

    # Example of calling the Simulated Annealing algorithm
    # best_solution_sa, best_cost_sa = simulated_annealing(coordinates)
    # print(f"Best Solution Cost (Simulated Annealing): {best_cost_sa}")

    # Example of calling the Genetic Algorithm
    # best_tour_ga, best_distance_ga = genetic_algorithm(coordinates, population_size=100, n_generations=100, mutation_rate=0.01, crossover_rate=0.9)
    # print(f"Best Distance (Genetic Algorithm): {best_distance_ga}")



    # while temp > stopping_temp:
    #     new_solution = generate_neighbor(list(current_solution))
    #     new_distance = total_distance(coords, new_solution)
    #
    #     if new_distance < best_distance:
    #         best_solution, best_distance = new_solution, new_distance
    #
    #     if new_distance < current_distance or math.exp((current_distance - new_distance) / temp) > random.random():
    #         current_solution, current_distance = new_solution, new_distance
    #
    #     temp *= cooling_rate
    #
    # return best_solution, best_distance

    # average_distance = np.mean(distances)
    # std_deviation = np.std(distances)

    # for run in range(num_runs):  # 30 independent runs
    #     iteration_distance = float('inf')
    #     for iteration in range(fitness_evaluations):  # 10,000 iterations simulated within each run
    #         print(f"Running iteration {iteration + 1}/{fitness_evaluations} for run {run + 1}/{num_runs} \r", end='')
    #         solution, distance = simulated_annealing(coords, temp, cooling_rate, stopping_temp)
    #         if distance < iteration_distance:
    #             iteration_distance = distance
    #             iteration_solution = solution
    #     distances.append(iteration_distance)
    #     if iteration_distance < best_overall_distance:
    #         best_overall_distance = iteration_distance
    #         best_overall_solution = iteration_solution

# Genetic Algorithm Functions
# def create_initial_population(size, n_cities):
#     return [random.sample(range(n_cities), n_cities) for _ in range(size)]
#
#
# def tournament_selection(population, scores, k=3):
#     selection_ix = np.random.randint(len(population), size=k)
#     selected_fitness = [scores[ix] for ix in selection_ix]
#     winner_ix = selection_ix[np.argmin(selected_fitness)]
#     return population[winner_ix]
#
#
# def crossover(parent1, parent2):
#     start, end = sorted(random.sample(range(len(parent1)), 2))
#     child = [None] * len(parent1)
#     child[start:end] = parent1[start:end]
#     pointer = end
#     for gene in parent2[end:] + parent2[:end]:
#         if gene not in child:
#             if pointer >= len(child):
#                 pointer = 0
#             child[pointer] = gene
#             pointer += 1
#     return child
#
#
# def mutate(tour, mutation_rate):
#     for i in range(len(tour)):
#         if random.random() < mutation_rate:
#             j = random.randint(0, len(tour) - 1)
#             tour[i], tour[j] = tour[j], tour[i]
#     return tour
#
#
# def genetic_algorithm(cities, population_size=100, n_generations=100, mutation_rate=0.01, crossover_rate=0.9):
#     # Create initial population
#     population = create_initial_population(population_size, len(cities))
#     best_distance = float('inf')
#     best_tour = None
#
#     for generation in range(n_generations):
#         # Evaluate fitness
#         scores = [total_distance(tour, cities) for tour in population]
#         for i, score in enumerate(scores):
#             if score < best_distance:
#                 best_distance, best_tour = score, population[i]
#
#         # Select parents
#         selected = [tournament_selection(population, scores) for _ in range(population_size)]
#
#         # Create the next generation
#         children = []
#         for i in range(0, population_size, 2):
#             parent1, parent2 = selected[i], selected[i + 1]
#             if random.random() < crossover_rate:
#                 child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
#                 children.append(mutate(child1, mutation_rate))
#                 children.append(mutate(child2, mutation_rate))
#             else:
#                 children.append(mutate(parent1.copy(), mutation_rate))
#                 children.append(mutate(parent2.copy(), mutation_rate))
#         population = children
#
#     return best_tour, best_distance
#
#
# def parameter_tuning_genetic_algorithm(cities, total_trials=100):
#     best_params = None
#     best_distance = float('inf')
#
#     # Example parameter ranges
#     population_sizes = [50, 100, 150]
#     mutation_rates = [0.01, 0.02, 0.05]
#     crossover_rates = [0.7, 0.85, 0.9]
#
#     # Generate all possible combinations of parameters
#     all_combinations = list(product(population_sizes, mutation_rates, crossover_rates))
#
#     # Randomly sample a subset of combinations if there are more than total_trials
#     if len(all_combinations) > total_trials:
#         sampled_combinations = random.sample(all_combinations, total_trials)
#     else:
#         sampled_combinations = all_combinations
#
#     # Iterate over the sampled combinations
#     for combo in sampled_combinations:
#         print(f"Running combination {combo}, best distance: {best_distance}", end='\r')
#         population_size, mutation_rate, crossover_rate = combo
#         distances = []
#         # Run a single trial for each combination due to total_trials constraint
#         tour, distance = genetic_algorithm(cities, population_size=population_size,
#                                            n_generations=50,  # Adjusted based on needs
#                                            mutation_rate=mutation_rate,
#                                            crossover_rate=crossover_rate)
#         distances.append(distance)
#         avg_distance = np.mean(distances)
#         if avg_distance < best_distance:
#             best_distance = avg_distance
#             best_params = combo
#
#     return best_params, best_distance
#
#
# def execute_with_tuned_parameters(cities, best_params, runs=30, fitness_evaluations=10000):
#     population_size, mutation_rate, crossover_rate = best_params
#     n_generations = fitness_evaluations // population_size
#
#     distances = []
#     for run in range(runs):
#         print(f"Running trial {run + 1}/{runs}", end='\r')
#         tour, distance = genetic_algorithm(cities, population_size=population_size,
#                                            n_generations=n_generations,
#                                            mutation_rate=mutation_rate,
#                                            crossover_rate=crossover_rate)
#         distances.append(distance)
#
#     return np.mean(distances), np.std(distances)
#
