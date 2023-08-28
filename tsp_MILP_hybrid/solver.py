#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import pulp

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_greedy(points):
    nodeCount = len(points)
    visited = [False] * nodeCount
    current_node = 0
    tour = [current_node]
    visited[current_node] = True
    
    for _ in range(nodeCount - 1):
        next_node = -1
        min_distance = float('inf')
        
        for candidate_node in range(nodeCount):
            if not visited[candidate_node]:
                distance = length(points[current_node], points[candidate_node])
                if distance < min_distance:
                    min_distance = distance
                    next_node = candidate_node
        
        if next_node != -1:
            tour.append(next_node)
            visited[next_node] = True
            current_node = next_node
    
    return tour

def nearest_neighbor(points):
    nodeCount = len(points)
    visited = [False] * nodeCount
    tour = [0]  # Start from the first city
    visited[0] = True
    
    for _ in range(nodeCount - 1):
        current_city = tour[-1]
        nearest_city = None
        min_distance = float('inf')
        
        for next_city in range(nodeCount):
            if not visited[next_city]:
                distance = length(points[current_city], points[next_city])
                if distance < min_distance:
                    nearest_city = next_city
                    min_distance = distance
        
        tour.append(nearest_city)
        visited[nearest_city] = True
        
    return tour

def solve_nearest_neighbor(points):
    initial_tour = nearest_neighbor(points)
    improved_tour = two_opt(initial_tour, points)
    
    return improved_tour

def two_opt(tour, points):
    n = len(tour)
    improved = True
    best_distance = calculate_tour_distance(tour, points)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - 1 == 1:
                    continue
                new_tour = tour[:]
                new_tour[i:j] = tour[j - 1: i - 1:-1]
                new_distance = calculate_tour_distance(new_tour, points)
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
    
    return tour

def calculate_tour_distance(tour, points):
    distance = 0
    n = len(tour)
    for i in range(n):
        current_city = tour[i]
        next_city = tour[(i + 1) % n]
        distance += length(points[current_city], points[next_city])
    return distance


def solve_local_search(points):
    nodeCount = len(points)
    initial_tour = list(range(nodeCount))

    # Apply 2 opt local search
    improved_tour = two_opt(initial_tour, points)

    return improved_tour


def solve_milp(points):
    nodeCount = len(points)

    # Create the PuLP problem instance
    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    # Create binary variables for decision variables
    x = [[pulp.LpVariable(f'x_{i}_{j}', cat=pulp.LpBinary) for j in range(nodeCount)] for i in range(nodeCount)]

    # Objective function: minimize the total distance
    obj = pulp.lpSum(x[i][j] * length(points[i], points[j]) for i in range(nodeCount) for j in range(nodeCount))
    prob += obj

    # Constraint 1: Each city is visited exactly once
    for i in range(nodeCount):
        prob += pulp.lpSum(x[i][j] for j in range(nodeCount)) == 1

    # Constraint 2: Each city is left exactly once
    for j in range(nodeCount):
        prob += pulp.lpSum(x[i][j] for i in range(nodeCount)) == 1

    # Subtour elimination constraints
    u = [pulp.LpVariable(f'u_{i}', lowBound=1, upBound=nodeCount, cat=pulp.LpInteger) for i in range(nodeCount)]
    for i in range(1, nodeCount):
        for j in range(1, nodeCount):
            if i != j:
                prob += u[i] - u[j] + nodeCount * x[i][j] <= nodeCount - 1

    # Solve the MILP problem
    prob.solve()

    # Extract the solution
    solution = []
    for i in range(nodeCount):
        for j in range(nodeCount):
            if pulp.value(x[i][j]) == 1:
                solution.append(j)

    return solution

def hybrid_algorithm(points):
    nodeCount = len(points)  

    # Choose the better solution between MILP and local search
    if nodeCount < 100:
        solution = solve_milp(points)
        length = calculate_tour_distance(solution, points)
        return solution, length
    elif 100 <= nodeCount < 300:
        solution = solve_local_search(points)
        length = calculate_tour_distance(solution, points)
        return solution, length
    elif 300 <= nodeCount < 900:
        solution = solve_nearest_neighbor(points)
        length = calculate_tour_distance(solution, points)
        return solution, length
    else:
        solution = solve_greedy(points)
        length = calculate_tour_distance(solution, points)
        return solution, length

def solve_it(input_data):
    # Parse the input
    lines = input_data.split('\n')
    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    solution = hybrid_algorithm(points)  # Get the solution from the hybrid algorithm
    obj = calculate_tour_distance(solution, points)  # Calculate the distance of the solution


    # Prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. Please select one from the data directory. (e.g., python solver.py ./data/tsp_51_1)')
