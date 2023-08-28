import math
import numpy as np
from collections import namedtuple
from itertools import product, combinations
from random import sample
from tqdm import tqdm

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def euclidean_distance(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)

def fits_capacity(v_tour, v_cap, customers):
    return sum(customers[c].demand for c in v_tour) <= v_cap

def tour_cost(v_tour, customers):
    return sum(euclidean_distance(customers[v_tour[i-1]], customers[v_tour[i]]) for i in range(len(v_tour)))

def place_customer(in_tour, customer, customers):
    best_tour = in_tour[:]
    best_tour.insert(1, customer)
    best_cost = tour_cost(best_tour, customers)

    for i in range(1, len(best_tour) - 1):
        new_tour = in_tour[:]
        new_tour.insert(i, customer)
        new_cost = tour_cost(new_tour, customers)
        if new_cost < best_cost:
            best_cost = new_cost
            best_tour = new_tour
    return best_tour

def objec(tours, customers):
    return sum(tour_cost(t, customers) for t in tours)

def idx_two_opt(sol, idx1, idx2):
    sol1 = sol[:idx1]
    sol2 = list(reversed(sol[idx1:idx2+1]))
    sol3 = sol[idx2+1:]
    new_sol = sol1 + sol2 + sol3
    return new_sol

def two_opt_search(tour, customers):
    nodeCount = len(tour)
    best_tour = tour
    best_obj = tour_cost(tour, customers)

    for i in range(1, nodeCount - 1):
        for j in range(1, nodeCount - 1):
            p_tour = idx_two_opt(best_tour, i, j)
            p_obj = tour_cost(p_tour, customers)
            if p_obj < best_obj:
                best_tour = p_tour
                best_obj = p_obj
    return best_tour

def four_swap(vehicle_tours, candidates, customers, v_cap):
    local_cpy_tours = [list(a) for a in vehicle_tours]
    start_obj = objec(vehicle_tours, customers)
    c_tours = []

    for c in candidates:
        for i, v_tour in enumerate(vehicle_tours):
            if c in v_tour:
                c_tours.append((c, i))
                break

    used_tours = set(c[1] for c in c_tours)

    for c in c_tours:
        local_cpy_tours[c[1]] = [i for i in local_cpy_tours[c[1]] if i != c[0]]

    best_obj = start_obj
    best_tours = [a for a in vehicle_tours]

    for perms in product(used_tours, repeat=4):
        cand_tours = local_cpy_tours[:]

        for i in range(4):
            cand_tours[perms[i]] = place_customer(cand_tours[perms[i]], candidates[i], customers)

        fits = all(fits_capacity(t, v_cap, customers) for t in cand_tours)

        if fits:
            cand_obj = objec(cand_tours, customers)

            if cand_obj < best_obj:
                best_obj = cand_obj
                best_tours = cand_tours.copy()

    return best_tours

def swap_solver(input_data):
    lines = input_data.split('\n')
    customer_count, vehicle_count, vehicle_capacity = map(int, lines[0].split())

    customers = []
    for i in range(1, customer_count + 1):
        parts = lines[i].split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    depot = customers[0]
    obj_hist = []
    best_obj = float('Inf')
    best_tours = []
    alpha_n = 50

    for alpha in tqdm(np.linspace(0, 1, alpha_n)):
        vehicle_tours = []
        remaining_customers = set(customers)
        remaining_customers.remove(depot)

        for v in range(0, vehicle_count):
            vehicle_tours.append([])
            capacity_remaining = vehicle_capacity
            current_c = 0
            full = False

            while remaining_customers and not full:
                sorted_rem_cust = sorted(remaining_customers,
                                         key=lambda x: alpha * (euclidean_distance(customers[current_c], x)) +
                                                       (1 - alpha) * (-x.demand), reverse=False)

                for c in sorted_rem_cust:
                    if c.demand <= capacity_remaining:
                        capacity_remaining -= c.demand
                        remaining_customers.remove(c)
                        current_c = c.index
                        vehicle_tours[v].append(c.index)
                        break

                full = all(c.demand > capacity_remaining for c in remaining_customers)

        valid = sum(len(v) for v in vehicle_tours) == len(customers) - 1

        if valid:
            for t in vehicle_tours:
                t.insert(0, 0)
                t.append(0)

            for i in range(len(vehicle_tours)):
                vehicle_tours[i] = two_opt_search(vehicle_tours[i], customers)

            obj = objec(vehicle_tours, customers)

            if obj < best_obj:
                best_obj = obj
                best_tours = vehicle_tours
                obj_hist.append(obj)
            else:
                obj_hist.append(obj)
        else:
            obj_hist.append(None)

    vehicle_tours = best_tours
    visited = [c for t in vehicle_tours for c in t]
    notvisited = [c for c in range(customer_count) if c not in visited]

    if notvisited:
        print('----- NOT ALL CLIENTS VISITED!!! ----- {} remain'.format(len(notvisited)))

    FITS = all(fits_capacity(t, vehicle_capacity, customers) for t in vehicle_tours)
    dontfit = [i for i, t in enumerate(vehicle_tours) if not fits_capacity(t, vehicle_capacity, customers)]

    print('All Fit?', FITS, 'dontfit:', dontfit)
    print('Starting cost = {}'.format(objec(best_tours, customers)))

    if customer_count < 100:
        maxit = 12200
    elif 100 <= customer_count < 200:
        maxit = 8200
    else:
        maxit = 5200

    swap_obj_hist = [objec(best_tours, customers)]

    for i in tqdm(range(maxit)):
        c_list = sample(range(1, customer_count), 4)
        vehicle_tours = four_swap(vehicle_tours, c_list, customers, vehicle_capacity)
        swap_obj_hist.append(objec(vehicle_tours, customers))

        if swap_obj_hist[-1] < swap_obj_hist[-2]:
            print('New best solution found on it {}. Obj: {}'.format(i + 1, swap_obj_hist[-1]))

    visited = [c for t in vehicle_tours for c in t]
    notvisited = [c for c in range(customer_count) if c not in visited]

    if notvisited:
        print('----- NOT ALL CLIENTS VISITED!!! ----- {} remain'.format(len(notvisited)))

    FITS = all(fits_capacity(t, vehicle_capacity, customers) for t in vehicle_tours)
    dontfit = [i for i, t in enumerate(vehicle_tours) if not fits_capacity(t, vehicle_capacity, customers)]

    print('All Fit?', FITS, 'dontfit:', dontfit)

    obj = objec(vehicle_tours, customers)
    print('Cost:', obj)

    outputData = '%.2f' % obj + ' ' + str(0) + '\n'

    for v in range(0, vehicle_count):
        outputData += ' '.join([str(customers[c].index) for c in vehicle_tours[v]]) + '\n'

    return outputData