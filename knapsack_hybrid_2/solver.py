#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def knapsack_branch_and_bound(items, capacity):
    n = len(items)
    items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
    
    upper_bound = 0
    best_solution = [0] * n
    stack = [(0, 0, 0, [])]  # (index, weight, value, taken)
    
    while stack:
        index, current_weight, current_value, taken = stack.pop()
        
        if index == n or current_weight > capacity:
            if current_value > upper_bound:
                upper_bound = current_value
                best_solution[:] = taken
            continue
        
        if current_weight + items[index].weight <= capacity:
            new_taken = taken.copy()
            new_taken.append(1)
            stack.append((index + 1, current_weight + items[index].weight, current_value + items[index].value, new_taken))
        stack.append((index + 1, current_weight, current_value, taken + [0]))

    return upper_bound, best_solution

def knapsack_greedy(items, capacity):
    n = len(items)
    ratios = [(items[i].value / items[i].weight, i) for i in range(n)]
    ratios.sort(reverse=True)

    total_value = 0
    taken = [0] * n
    weight = 0

    for _, index in ratios:
        if weight + items[index].weight <= capacity:
            taken[index] = 1
            total_value += items[index].value
            weight += items[index].weight
    
    return total_value, taken

def knapsack_dynamic_programming(items, capacity):
    '''
    Solves the knapsack problem optimally using dynamic programming.

    Args:
        items: A list of named tuples representing items (index, value, weight)
        capacity: The total weight capacity of the knapsack.
    
    Returns:
        A tuple containing the total value of the optimal solution and a list of
        decision variables (0 or 1) indicating whether each item is taken or not.
    '''
    # Define the number of items
    n = len(items)
    # Create a dynamic programming table dp with (n+1) rows and (capacity+1) columns
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Populate the dp table using bottom-up dynamic programming
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # If the current item's weight is less than or equal to the current capacity
            if items[i - 1].weight <= w:
                # Choose the maximum value between not taking the current item and taking it
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - items[i - 1].weight] + items[i - 1].value)
            else:
                # If the current item's weight exceeds the current capacity, do not take it
                dp[i][w] = dp[i - 1][w]
    
    # The optimal solution value is stored in the bottom-right cell of the dp table
    total_value = dp[n][capacity]
    taken = [0] * n
    w = capacity

    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            taken[i - 1] = 1
            w -= items[i - 1].weight
    
    return total_value, taken

def knapsack_hybrid(items, capacity):
    n = len(items)
    
    # Choose a threshold to switch from branch and bound, to dynamic programming to greedy
    threshold_1 = 1000000
    threshold_2 = 1000001
    threshold_3 = 10000000
    
    if n * capacity < threshold_1:
        # Use branch and bound for small instances
        return knapsack_branch_and_bound(items, capacity)
    elif threshold_2 < n * capacity < threshold_3:
        # Use dynamic programming for medium instances
        return knapsack_dynamic_programming(items, capacity)
    else:
        # Use greedy for larger instances
        return knapsack_greedy(items, capacity)

def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # call the hybrid algorithm
    value, taken = knapsack_hybrid(items, capacity)
        
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
