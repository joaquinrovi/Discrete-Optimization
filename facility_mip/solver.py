#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import pulp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # Create a MIP problem
    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)

    # Create binary decision variables
    a = [pulp.LpVariable("a{}".format(f.index), cat = pulp.LpBinary) for f in facilities]
    b = [[pulp.LpVariable("b{}_{}".format(f.index, c.index), cat=pulp.LpBinary) for c in customers] for f in facilities]

    # Objective function
    prob += pulp.lpSum(f.setup_cost * a[f.index] for f in facilities) + \
            pulp.lpSum(length(f.location, c.location) * b[f.index][c.index] for f in facilities for c in customers)

    # Constraints
    for c in customers:
        prob += pulp.lpSum(b[f.index][c.index] for f in facilities) == 1
    
    for f in facilities:
        prob += pulp.lpSum(c.demand * b[f.index][c.index] for c in customers) <= f.capacity

    for f in facilities:
        for c in customers:
            prob += b[f.index][c.index] <= a[f.index]


    # Solve the model
    time = 300 # 300 seconds time limit
    gurobi_solver = pulp.GUROBI_CMD(timeLimit = time)
    prob.solve(solver = gurobi_solver)

    # Extract the solution
    solution = [-1] * len(customers)
    for c in customers:
        for f in facilities:
            if pulp.value(b[f.index][c.index]) == 1:
                solution[c.index] = f.index
                break
    
    used = [0] * len(facilities)
    for f in solution:
        used[f] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

