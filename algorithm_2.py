import random
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def genetic_algorithm(tasks, UAVs):
    # Define Genetic Algorithm Parameters
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 100
    MUTATION_RATE = 0.1

    def execute_task(task, UAV):
        UAV["location x"] = task["location x"]
        UAV["location y"] = task["location y"]

    def calculate_time(task, UAV, current_position):
        travel_time = calculate_travel_time(task, UAV, current_position)
        task_time = task["tasktime(min)"]
        return travel_time + task_time

    def calculate_travel_time(task, UAV, current_position):
        x_diff = abs(task["location x"] - current_position[0])  # Use Current Position Coordinates
        y_diff = abs(task["location y"] - current_position[1])  # Use Current Position Coordinates
        distance = (x_diff ** 2 + y_diff ** 2) ** 0.5
        travel_time = distance / UAV["speed"]
        return travel_time

    def calculate_satisfaction(task_time, task_deadline):
        return -task_time / task_deadline + 1

    def is_gene_valid(gene, task_index):
        task_type = tasks[task_index]["type"]
        UAV_type = UAVs[gene - 1]["type"]
        if (task_type == 'O' and UAV_type in ['X', 'Y', 'Z']) or \
           (task_type == 'C' and UAV_type in ['Y', 'Z']) or \
           (task_type == 'G' and UAV_type in ['X', 'Y']) or \
           (task_type == 'F' and UAV_type in ['Z']):
            return True
        else:
            return False

    def generate_initial_population():
        population = []
        for _ in range(POPULATION_SIZE):
            chromosome = []
            for task_idx in range(len(tasks)):
                gene = random.randint(1, len(UAVs))  # Represents the UAV by which the task is performed
                while not is_gene_valid(gene, task_idx):  # Check that the genes fulfil the matching requirements
                    gene = random.randint(1, len(UAVs))
                chromosome.append(gene)
            population.append(chromosome)
        return population

    def calculate_fitness(chromosome):
        total_satisfaction = 0
        UAV_times = {UAV["NO"]: 0 for UAV in UAVs}  # Store the time of completed tasks for each UAV
        UAV_positions = {UAV["NO"]: (UAV["location x"], UAV["location y"]) for UAV in UAVs}  # Store the current coordinates of each UAV
        for i, gene in enumerate(chromosome):
            UAV = UAVs[gene - 1]
            task = tasks[i]
            task_time = calculate_time(task, UAV, UAV_positions[UAV["NO"]])  # Pass in the current UAV coordinates
            UAV_times[UAV["NO"]] += task_time  # Update the time of completed tasks for this UAV
            UAV_positions[UAV["NO"]] = (task["location x"], task["location y"])  # Update the current coordinates of the UAV
            satisfaction = calculate_satisfaction(UAV_times[UAV["NO"]], task["deadline"])
            total_satisfaction += satisfaction
        return total_satisfaction

    def selection(population):
        return sorted(population, key=calculate_fitness, reverse=True)[:POPULATION_SIZE]

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutation(chromosome):
        mutated_chromosome = chromosome[:]
        for i in range(len(mutated_chromosome)):
            if random.random() < MUTATION_RATE:
                mutated_chromosome[i] = random.randint(1, len(UAVs))
                while not is_gene_valid(mutated_chromosome[i], i):
                    mutated_chromosome[i] = random.randint(1, len(UAVs))
        return mutated_chromosome

    def elitism(population):
        return selection(population)

    def termination_condition(population, generation):
        return generation >= MAX_GENERATIONS

    population = generate_initial_population()
    fitness_history = []  # List to store the fitness values of the best chromosome in each generation

    for generation in range(MAX_GENERATIONS):
        population = selection(population)
        selected_population = population[:POPULATION_SIZE]

        children = []
        while len(children) < POPULATION_SIZE:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            children.append(child)

        population = elitism(selected_population + children)

        if termination_condition(population, generation):
            break

        best_chromosome = population[0]
        fitness = calculate_fitness(best_chromosome)
        fitness_history.append(fitness)
        
    return best_chromosome, fitness_history

plt.style.use("ggplot")

def plot_flight_paths(chromosome, UAVs, tasks):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colors for different UAVs

    # Initialize task locations and UAV starting positions
    task_locations = [(task["location x"], task["location y"]) for task in tasks]
    uav_positions = [(0, 0) for _ in UAVs]  # Set the starting position for each UAV to (0, 0)

    # Plot task locations with distinct markers and colors
    ax.scatter(*zip(*task_locations), marker='x', color='red', s=100, label='Tasks')
    ax.scatter(*zip(*uav_positions), marker='o', color='black', s=120, label='UAV Start')

    # Initialize plot elements for each UAV path with unique colors and thicker lines
    uav_paths = [[(0, 0)] for _ in range(len(UAVs))]  # Each path starts at (0, 0)
    uav_lines = [ax.plot([], [], 'o-', color=colors[i % len(colors)], linewidth=2, markersize=6)[0] for i in range(len(UAVs))]

    # Set plot limits based on the range of task locations, adding some padding
    ax.set_xlim(0, max([pos[0] for pos in task_locations]) + 50)
    ax.set_ylim(0, max([pos[1] for pos in task_locations]) + 50)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("UAV Task Assignment and Paths")

    # Add a legend for clarity
    ax.legend(loc="upper right")

    # Initialize function for animation
    def init():
        for line in uav_lines:
            line.set_data([], [])
        return uav_lines

    # Update function for each frame
    def update(frame):
        for i, gene in enumerate(chromosome[:frame + 1]):  # Process each task up to the current frame
            uav_idx = gene - 1
            task = tasks[i]
            uav_paths[uav_idx].append((task["location x"], task["location y"]))
            uav_lines[uav_idx].set_data(*zip(*uav_paths[uav_idx]))  # Update line data for each UAV
        return uav_lines

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(tasks), init_func=init, interval=500, blit=True)

    # Save animation as gif
    anim.save("flight_paths.gif", writer='imagemagick')
    plt.close(fig)  # Close the plot to prevent display in Streamlit

    # Display the animation in Streamlit
    st.subheader("Animated Flight Paths")
    st.image("flight_paths.gif")  # Display the animated gif in Streamlit

