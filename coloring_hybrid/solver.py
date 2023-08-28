#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from ortools.sat.python import cp_model
import random
import math
import time

def greedy_coloring(graph):
    n = len(graph)
    colors = [-1] * n
    max_color = -1
    
    for node in range(n):
        used_colors = set()
        for neighbor in graph[node]:
            if colors[neighbor] != -1:
                used_colors.add(colors[neighbor])
        
        for color in range(n):
            if color not in used_colors:
                colors[node] = color
                max_color = max(max_color, color)
                break
    
    return max_color, colors

from ortools.sat.python import cp_model
import time

def cp_coloring(graph, time_limit=60):
    n = len(graph)
    model = cp_model.CpModel()
    colors = [model.NewIntVar(0, n - 1, f'colors_{i}') for i in range(n)]

    for node in range(n):
        for neighbor in graph[node]:
            model.Add(colors[node] != colors[neighbor])

    max_color_var = model.NewIntVar(0, n - 1, 'max_color')
    model.AddMaxEquality(max_color_var, colors)

    model.Minimize(max_color_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit  # Set the time limit

    start_time = time.time()
    status = solver.Solve(model)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        max_color = solver.Value(max_color_var)
        node_colors = [solver.Value(c) for c in colors]
        return max_color, node_colors, elapsed_time
    else:
        return -1, [], elapsed_time


def local_search_coloring(graph, max_iterations=10000):
    n = len(graph)
    current_solution = greedy_coloring(graph)[1]
    current_max_color = max(current_solution)
    
    for _ in range(max_iterations):
        node_to_change = random.randint(0, n - 1)
        current_color = current_solution[node_to_change]
        new_color = current_color
        
        while new_color == current_color:
            new_color = random.randint(0, current_max_color)
        
        current_solution[node_to_change] = new_color
        
        # Check if the new color creates conflicts with adjacent nodes
        has_conflict = any(current_solution[neighbor] == new_color for neighbor in graph[node_to_change])
        
        if has_conflict:
            current_solution[node_to_change] = current_color
    
    return max(current_solution), current_solution

def simulated_annealing_coloring(graph, max_iterations=20000, initial_temperature=100.0, cooling_rate=0.95):
    n = len(graph)
    current_solution = greedy_coloring(graph)[1]
    current_max_color = max(current_solution)

    best_solution = current_solution[:]
    best_max_color = current_max_color

    current_temperature = initial_temperature

    for _ in range(max_iterations):
        node_to_change = random.randint(0, n - 1)
        new_color = random.randint(0, current_max_color)
        
        current_solution[node_to_change] = new_color
        new_max_color = max(current_solution)
        
        if new_max_color < best_max_color:
            best_solution = current_solution[:]
            best_max_color = new_max_color
        else:
            delta = new_max_color - best_max_color
            acceptance_prob = math.exp(-delta / current_temperature)
            if random.random() < acceptance_prob:
                best_solution = current_solution[:]
                best_max_color = new_max_color
        
        current_temperature *= cooling_rate
    
    return best_max_color, best_solution

def solve_it(input_data):
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    graph = [[] for _ in range(node_count)]
    for edge in edges:
        u, v = edge
        graph[u].append(v)
        graph[v].append(u)

    if node_count > 10000: # Choose a threshold for the hybrid approach
        max_color, node_colors = greedy_coloring(graph)
    elif 5000 < node_count <= 10000:
        max_color, node_colors = simulated_annealing_coloring(graph)
    elif 500 < node_count <= 5000:
        max_color, node_colors = local_search_coloring(graph)
    else:
        max_color, node_colors, _ = cp_coloring(graph)

    output_data = f"{max_color+1} 0\n"
    output_data += " ".join(map(str, node_colors))

    return output_data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
