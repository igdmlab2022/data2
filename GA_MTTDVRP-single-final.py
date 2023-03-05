import argparse
import torch
import math
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import sys
from plot import get_clean_path, get_routes_from_list, plot_route
import torch
import pickle
import argparse
import torch
import math
from math import degrees, atan2, sqrt
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import pandas
import geopy.distance
from time import time
# from data import generate_data, data_from_Solomon
from plot import get_clean_path, get_routes_from_list, get_mt_hf_routes_from_list, plot_route, plot_route_geomap

CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.
}

def get_dist(matrix, src, dst, tp):
    # matrix = data['tdtt']
    return matrix[src, dst, tp]

def get_prob():
    return random.random()

def get_fitness(sol, matrix, demands, capacity):

    li = copy.deepcopy(sol) # without it, new_list becomes identical with li having depots at first and last
    if li[0] != 0: # if the first is not depot
        li.insert(0,0) # add depot at first
    if li[-1] != 0: # if the last is not depot
        li.append(0) # add depot at last

    fitness = 0
    vehicle_time = 0

    for i in range(len(li)-1):
        if vehicle_time < 240:
            dist = get_dist(matrix, li[i], li[i+1], 0)
        elif vehicle_time < 480:
            dist = get_dist(matrix, li[i], li[i+1], 1)
        elif vehicle_time <= 720:
            dist = get_dist(matrix, li[i], li[i+1], 2)
        else:
            fitness = math.inf
            break
        fitness += dist # get_dist(nodes[i].xy, nodes[i+1].xy)
        vehicle_time += dist
        if li[i+1] == 0:
            vehicle_time = 0

    # if it's not valid route, then set inf for fitness
    temp = copy.deepcopy(li)
    curr_demand = 0
    for i in range(len(temp)):
        if temp[i] == 0 and curr_demand > capacity: # 0 represents depot
            fitness = math.inf
            return fitness
        elif temp[i] == 0:
            curr_demand = 0
        else:
            curr_demand += demands[temp[i]]

    return fitness

def get_initial_population(pop_size, li, matrix, demands, capacity):

    population = []

    while len(population) < pop_size: #for i in range(pop_size):
        # generate new list (solution)
        temp_list = copy.deepcopy(li)
        new_list = [] # new_list is like chromosome in GA
        while(len(temp_list) > 0):
            index = (int)(get_prob() * len(temp_list)) # i.e. 12
            new_list.append(temp_list.pop(index))

        new_fit = get_fitness(new_list, matrix, demands, capacity)

        if new_fit != math.inf:
            population.append(IndividualSolution())
            population[-1].set_solution(new_list)
            population[-1].set_fitness(new_fit)

    return population

def create_new_sol(li, matrix, demands, capacity):
    # input: idx_list

    while 1:
        temp_list = copy.deepcopy(li)
        new_list = []  # new_list is like chromosome in GA
        while (len(temp_list) > 0):
            index = (int)(get_prob() * len(temp_list))  # i.e. 12
            new_list.append(temp_list.pop(index))

        new_fit = get_fitness(new_list, matrix, demands, capacity)
        if new_fit != math.inf:
            new_sol = IndividualSolution()
            new_sol.set_solution(new_list)
            new_sol.set_fitness(new_fit)
            break

    return new_sol

def selection(population, k): # tournament selection
    n = 2 # n is set to 2 for two parents
    selected = []
    sel_indices = []
    for _ in range(n): # n number of tournaments
        indices = [random.randint(0, len(population)-1) for _ in range(k)]
        k_tournament = [population[ind].fitness for ind in indices]
        _, minidx_k = min((val, idx) for (idx, val) in enumerate(k_tournament))
        selected.append(population[indices[minidx_k]])
        sel_indices.append(indices[minidx_k])

    return selected, sel_indices # list of size n

def crossover(parent1, parent2, matrix, demands, capacity):
    # parent1: [2, 8, 18, 12, 0, 3, 14, 13, 9, 16, 7, 0, 1, 20, 17, 10, 5, 15, 0, 4, 6, 11, 19]
    # parent2: [4, 17, 0, 2, 3, 18, 16, 6, 14, 0, 19, 10, 9, 8, 20, 7, 0, 15, 13, 1, 5, 11, 12]

    left = random.randint(1, len(parent1.solution) - 2) # i.e. 4
    right = random.randint(left, len(parent1.solution) - 1) # i.e. 10

    c1 = []  # c1 = [c for c in parent1.solution[0:] if c not in parent2.solution[left:right + 1]]
    temp = parent2.solution[left:right + 1] # parent2.solution[left:right + 1]: [3, 18, 16, 6, 14, 0, 19]
    for c in parent1.solution[0:]:
        if c not in temp:
            c1.append(c)
        elif c in temp:
            if c == 0: # this is because depot(0) is included multiple times in the list
                removeidx = temp.index(0)
                temp.pop(removeidx) # remove that one zero from temp
        # c1: [2, 8, 12, 13, 9, 7, 0, 1, 20, 17, 10, 5, 15, 0, 4, 11]
    child1_sol = c1[:left] + parent2.solution[left:right + 1] + c1[left:]
        # child1: [2, 8, 12, 13, 3, 18, 16, 6, 14, 0, 19, 9, 7, 0, 1, 20, 17, 10, 5, 15, 0, 4, 11]

    c2 = []  # c2 = [c for c in parent2.solution[0:] if c not in parent1.solution[left:right + 1]]
    temp = parent1.solution[left:right + 1]
    for c in parent2.solution[0:]:
        if c not in temp:
            c2.append(c)
        elif c in temp:
            if c == 0:
                removeidx = temp.index(0)
                temp.pop(removeidx)
        # c2: [4, 17, 2, 18, 6, 0, 19, 10, 8, 20, 0, 15, 1, 5, 11, 12]
    child2_sol = c2[:left] + parent1.solution[left:right + 1] + c2[left:]
        # child2: [4, 17, 2, 18, 0, 3, 14, 13, 9, 16, 7, 6, 0, 19, 10, 8, 20, 0, 15, 1, 5, 11, 12]

    # feasibility check (if it's not feasible, return the parent)
    child1_fit = get_fitness(child1_sol, matrix, demands, capacity)
    child2_fit = get_fitness(child2_sol, matrix, demands, capacity)

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    if child1_fit != math.inf:
        child1.set_solution(child1_sol)
        child1.set_fitness(child1_fit)
    if child2_fit != math.inf:
        child2.set_solution(child2_sol)
        child2.set_fitness(child2_fit)

    return child1, child2

def mutation(sol, matrix, demands, capacity):

    mut = copy.deepcopy(sol)
    mut_sol = mut.solution

    left = random.randint(1, len(mut_sol) - 2)
    right = random.randint(left, len(mut_sol) - 1)
    mut_sol[left], mut_sol[right] = mut_sol[right], mut_sol[left]

    # feasibility check will be done in main
    mut_fit = get_fitness(mut_sol, matrix, demands, capacity)

    mut.set_solution(mut_sol)
    mut.set_fitness(mut_fit)

    return mut

class IndividualSolution:

    def __init__(self):
        self.solution = None
        self.fitness = math.inf

    def set_solution(self, solution):
        self.solution = solution

    def set_fitness(self, fitness):
        self.fitness = fitness


def get_tp(curr_time):
    # print("curr_time", curr_time)
    if curr_time < 240:
        return 0
    elif curr_time < 480:
        return 1
    else:
        return 2
    

def get_my_mt_hf_routes_from_list(matrix, pi_):
    cost = 0
    curr_time = 0 
    max_hour = 720
    vehicles = 1
    last = 0
    for node in pi_:
        if node == 0 and (curr_time / max_hour >= 3/4) and (np.random.uniform(0., 1.) >= (1 - curr_time / max_hour)):
            cost += get_dist(matrix, last, node, get_tp(curr_time))
            vehicles += 1
            curr_time = 0
            last = node
        else:
            cost += get_dist(matrix, last, node, get_tp(curr_time))
            curr_time += get_dist(matrix, last, node, get_tp(curr_time))
            last = node
    return cost, vehicles


def GA(loc, matrix, demands, capacity):
    t1 = time()
    # list of indices where depot index (0) is added by the NUM_vehicles-1
    idx_list = [i for i in range(1,len(loc))] # 1, 2, ..., n_customer
    idx_list.extend(0 for _ in range(NUM_vehicles-1))

    # generate initial population
    population = get_initial_population(POPULATION_size, idx_list, matrix, demands, capacity)

    # perform evolution
    best = IndividualSolution()
    minfit_by_itr = [] # for plotting
    for g in range(NUM_generation):

        for _ in range(int(POPULATION_size/2)):
            # selection
            selected, sel_indices = selection(population, 2) # n=2 (two tournaments), k=2 (two chromosome in each tournament)
            parent1 = selected[0] # [5, 4, 19, 13, 11, 15, 0, 8, 20, 18, 12, 16, 9, 0, 17, 10, 14, 1, 0, 2, 3, 6, 7]
            parent2 = selected[1] # [11, 17, 18, 3, 0, 19, 4, 10, 13, 0, 12, 7, 16, 14, 6, 20, 15, 0, 2, 5, 9, 8, 1]
            nsame = sum(l1 == l2 for (l1, l2) in zip(parent1.solution, parent2.solution)) # number of identical occurrence

            # crossover
            if get_prob() < CROSSOVER_rate:
                child1, child2 = crossover(parent1, parent2, matrix, demands, capacity)

                # mutation
                if nsame/len(idx_list) > MUTATION_rate: # get_prob() < MUTATION_rate:
                    child1 = mutation(child1, matrix, demands, capacity)
                    if child1.fitness == math.inf: # if the mutated child is infeasible
                        child1 = create_new_sol(idx_list, matrix, demands, capacity)
                    child2 = mutation(child2, matrix, demands, capacity)
                    if child2.fitness == math.inf: # if the mutated child is infeasible
                        child2 = create_new_sol(idx_list, matrix, demands, capacity)

                # remove two worst sols in population and add two childs
                population.sort(key=lambda x: x.fitness, reverse=False) # ascending order
                del population[-2:]
                population.append(child1)
                population.append(child2)


        # best solution by generation
        curr_min = min(population, key=lambda x: x.fitness)
        minfit_by_itr.append(curr_min.fitness) # for plot
        if curr_min.fitness < best.fitness:
            best.set_solution(curr_min.solution)
            best.set_fitness(curr_min.fitness)
        # print('min cost of #', g, ': ', curr_min.fitness)

    # depot is added at first and last in a best list
    pi_ = get_clean_path(best.solution)
    res = get_my_mt_hf_routes_from_list(matrix, pi_)
    # routes, _ = get_routes_from_list(pi_)

    # print('---------')
    # print('Total cost: ', res, ', Routes: ', pi_)
    # print(f'run time:{time() - t1}s')

    return res, (time() - t1)

def solve_instance(data):
# calling ACO
# preliminary step
    depot_xy = data['depot'].cpu().numpy()  # np (2,) i.e. [0.29611194, 0.5165623 ]
    # print("depot_xy loaded")
    customer_xy = data['loc'].cpu().numpy()  # np (n_customer,2)
    # print("customer_xy loaded")
    demands = data['demand'].cpu().numpy()  # np (n_customer,) i.e.[0.26666668, 0.26666668, 0.3       , 0.26666668, ..]
    demands = np.append(0, demands) # adding depot demand as 0 -> now np (n_customer+1,)
    # print("demands loaded")
    xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis=0)  # np (n_customer+1, 2)
    #matrix = data['tdtt2'].cpu().numpy()  # np (n_customer,n_customer, n_time_period)
    # print("matrix loaded")
    try:
        matrix = data['tdtt2'].cpu().numpy()  # np (n_customer,n_customer, n_time_period)
    except:
        matrix = data['tdtt'].cpu().numpy()  # np (n_customer,n_customer, n_time_period)

    n_customer = len(customer_xy)
    # print("n_customer loaded")

    CAPACITIES = 1 #CAPACITIES[n_customer]

    # dist = get_dist_mat(xy) # distance matrix including depot

    return GA(xy, matrix, demands, CAPACITIES)

POPULATION_size = 1000
NUM_generation = 500
NUM_vehicles = 4
MUTATION_rate = 0.1 # mutation rate
CROSSOVER_rate = 0.9
num_samples = 10000
max_hour = 720

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', metavar='SE', type=int, default=1234,
                        help='random seed number for inference, reproducibility')
    parser.add_argument('-n', '--n_customer', metavar='N', type=int, default=20,
                        help='number of customer nodes, time sequence')
    parser.add_argument('--test_size', type=int, default=1, help='number of test problems')

    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    size = args.n_customer
    with open('VRP' + str(size) + '.pkl', 'rb') as handle:
        data = pickle.load(handle)
    result = 0
    run_time = 0
    iteration = 0
    out_file = 'log_GA_VRP' + str(size) + '-' + str(time()) + '.txt'
    if size == 10:
        POPULATION_size = 100
        NUM_generation = 200
        NUM_vehicles = 4
    elif size == 20:
        POPULATION_size = 200
        NUM_generation = 500
        NUM_vehicles = 5
    elif size == 50:
        POPULATION_size = 2000
        NUM_generation = 1000
        NUM_vehicles = 5
    elif size == 100:
        POPULATION_size = 5000
        NUM_generation = 2000
        NUM_vehicles = 5
    for datum in data:
        res = solve_instance(datum)
        result += res[0]
        run_time += res[1]
        iteration += 1
        with open(out_file, 'a') as handle:
            handle.writelines(["Iteration:" + str(iteration), 
                               "\n Cost:" + str(res[0]), 
                               "\n Run Time:" + str(res[1]), 
                               "\n Total Cost:" + str(result), 
                               "\n Total Run Time:" + str(run_time), 
                               "\n Avg Cost:" + str(result / iteration),
                               "\n Avg Time:" + str(run_time / iteration),

                               "\n ----- \n \n"])
            handle.close()

    with open(out_file, 'a') as handle:
        handle.writelines(["\n ------------------------------- \n \n",
                            "\n Total Iterations:" + str(iteration), 
                            "\n Final Total Cost:" + str(result), 
                            "\n Final Total Time:" + str(run_time), 
                            "\n Avg Cost:" + str(result / iteration), 
                            "\n Avg Time:" + str(run_time / iteration), 
                            "\n ----- \n \n"])
        handle.close()

