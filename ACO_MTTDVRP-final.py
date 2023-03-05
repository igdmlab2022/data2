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

# constants
NUM_ants = 100 # number of ants (which is colony size just like population size in GA)
# NUM_iterations = 500 # number of iterations
EVP_rate = 0.1 # evaporation rate
# size = 10
# num_samples = 10000
max_hour = 720

CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.
}

def get_dist(matrix, src, dst, tp):
    # matrix = data['tdtt']
    return matrix[src, dst, tp]

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
        else:
            dist = get_dist(matrix, li[i], li[i+1], 2)
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

class IndividualSolution:

    def __init__(self):
        self.solution = None
        self.fitness = math.inf
        # self.service = None # service range in time
        # self.twcost = None

        self.vehidx = None # for multi-trip feature, because one vehicle can take multiple trips. Starting from 0
        # self.vehtype = None # for heterogenous vehicle type feature. Can have a value from [0, 1, 2, 3] where 0 is no preference customer

    def set_solution(self, solution):
        self.solution = solution

    def set_fitness(self, fitness):
        self.fitness = fitness

    # def set_service(self, service):
    #     self.service = service

    # def set_twcost(self, twcost):
    #     self.twcost = twcost

    def set_vehidx(self, vehidx):
        self.vehidx = vehidx

    # def set_vehtype(self, vehtype):
    #     self.vehtype = vehtype


class Env():
    def __init__(self, matrix): # all ndarray (matrix)
        size = len(matrix) - 1
        # initial pheromone
        self.tau = np.ones([size+1, size+1]) # initialize the pheromone as 1
        np.fill_diagonal(self.tau, 0) # set diagonal values to 0

        # initial etta (visibility)
        # print(dist.shape)
        dist_inf = copy.deepcopy(matrix[:, :, 0]) # diagonal value in dist is 0 -> set to Inf
        np.fill_diagonal(dist_inf, math.inf)
        self.etta = 1/dist_inf
        b = 1.0 # # normalization in range (a, b)
        a = 0.0
        self.etta = a + ((self.etta - np.min(self.etta)) / (np.max(self.etta) - np.min(self.etta))) * (b - a)

        # initial probability
        self.prob = np.zeros([size+1, size+1])
        self.update_prob()

    def update_pheromone(self, colony):
        self.tau = (1-EVP_rate) * self.tau # evaporate pheromone
        for k in range(NUM_ants):
            # put pheromone based on the whole solution
            amount_phr = (1 / colony[k].fitness) * size
            for arc in range(len(colony[k].solution)-1):
                i = colony[k].solution[arc]
                j = colony[k].solution[arc+1]
                self.tau[i,j] = self.tau[i,j] + amount_phr # deposit the pheromone. + EVP_rate * amount_phr

        self.tau[0,0] = 0
        return self.tau

    def update_prob(self):
        for i in range(size+1):
            denominator = sum(self.tau[i,:] * self.etta[i,:])
            for j in range(i,size+1):
                self.prob[i,j] = self.tau[i,j] * self.etta[i,j] / denominator
                self.prob[j,i] = self.prob[i,j] # copy the value to lower matrix
        return self.prob
    

def get_tp(curr_time):
    # print("curr_time", curr_time)
    if curr_time < 240:
        return 0
    elif curr_time < 480:
        return 1
    else:
        return 2
    

def ant_movement(env, matrix, demands, capacity):
    size = len(matrix) - 1
    li = []
    li_vidx = []
    # li_vtype = []
    temp = list(range(1, size + 1))
    visit = []
    curr_load = 0  # current cumulative demands (weight and skids)
    curr_loc = 0  # current location of ant (depot at the begining)
    curr_time = 0 # current cumulative time
    curr_vidx = 0 # current vehicle index (starting from 0)
    # curr_vtype = 0 # current vehicle type

    while len(temp) != 0:
        prob_temp = copy.deepcopy(env.prob[curr_loc, :])  # np (n_customer+1,)

        weighted_prob = 0  # flag whether prob is weighted for strictly restricted customers
        prob_temp[visit] = 0 # set the prob of visited cities as 0
        prob_temp[0] = 0 # set the prob of depot as 0

        for i in range(len(prob_temp)):
            current_tp = get_tp(curr_time)
            next_time = current_tp + get_dist(matrix, curr_loc, i, current_tp)
            next_tp = get_tp(next_time)
            if next_time+ get_dist(matrix, i, 0, next_tp) > max_hour:
                prob_temp[i]=0

        # prob_temp[curr_time + timemat[curr_loc, :] + timemat[:, 0] > max_hour] = 0 # set the prob of customers to 0 if they're over the maximum working hours

        next = random.choices(temp, weights=tuple(prob_temp[temp]), k=1)[0] # get the next node from curr_loc based on prob

        early_change = 0
        if curr_loc == 0: # if the vehicle is currently at depot (completed a trip)
           if (curr_time / max_hour >= 3/4) and (np.random.uniform(0., 1.) >= 1 - curr_time / max_hour): # np.random.uniform(3/4, 1.0) < curr_time / max_hour: # if random number from uniform distribution is smaller than the working-hour-usage of the current vehicle
               early_change = 1 # flag is set to 1

        if (sum(prob_temp) == 0) or (early_change == 1): # there's no customer to visit by this vehicle or veh_change_flag is 1, head to depot

            next = 0  # next node is depot
            curr_load = 0  # initialize the cumulative demand

            if curr_loc == 0: # 2nd situation (must-change) for vehicle change
                curr_vidx += 1  # new vehicle
                # curr_vtype = 0  # initialize the current vehicle type
                curr_time = 0  # initialize the current cumulative time

        else:

            next = random.choices(temp, weights=tuple(prob_temp[temp]), k=1)[0] # get the next node from curr_loc based on prob

            # if (curr_load[0] + demands[next, 0] <= CAPA_WEIGHT[max(curr_vtype, vr[next])] and curr_load[1] + demands[next, 1] <= CAPA_SKID[max(curr_vtype, vr[next])]): # MT, CAPA, TW(lt) constraints
            if curr_load + demands[next] <= capacity:
                # if curr_vtype == 0: # if the vehicle type is not yet specified, set the vehicle type
                # curr_vtype = vr[next]

                temp.remove(next)
                visit.append(next)
                curr_load += demands[next]
                current_tp = get_tp(curr_time)
                curr_time += get_dist(matrix, curr_loc, next, current_tp)
                # if curr_loc == 0: # if this trip just got started from depot, add loading time occurred at depot
                #     curr_time += st[curr_loc]

            else: # heading to depot

                # if curr_vtype == 0:  # if the vehicle type is still 0 even when a trip is made,
                #     curr_vtype = 1 # set it to 1 to prevent that this vehicle is set to be smaller truck at subsequent multi-trips
                #     if li_vtype[-1] == 0: # update the the most li_vtype having 1 for curr_vtype if it's 0
                #         li_vtype[-1] = curr_vtype

                next = 0  # next node is depot
                curr_load = 0  # initialize the cumulative demand
                current_tp = get_tp(curr_time)
                curr_time += get_dist(matrix, curr_loc, 0, current_tp)

        li.append(next)
        li_vidx.append(curr_vidx)
        # li_vtype.append(curr_vtype)
        curr_loc = next


        # if curr_load + demands[next] <= capacity:  # capacity constraint
        #     temp.remove(next)
        #     visit.append(next)
        #     curr_load += demands[next]
        # else:
        #     next = 0  # next node is depot
        #     curr_load = 0  # initialize the cumulative demand

        # li.append(next)
        # curr_loc = next

    # return li # ant's route
    return li, li_vidx

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

def ACO(loc, matrix, demands, capacity):

    t1 = time()

    env = Env(matrix)

    # initialize population (colony)
    colony = []
    for k in range(NUM_ants):
        colony.append(IndividualSolution())

    tau = []
    prob = []

    best = IndividualSolution()
    minfit_by_itr = [] # for plotting
    for t in range(NUM_iterations):

        for k in range(NUM_ants):
            # ant moving
            sol, vidx = ant_movement(env, matrix, demands, capacity)
            fit = get_fitness(sol, matrix, demands, capacity) # get a fitness

            # rewrite-the-solution approach
            #colony[k].set_solution(sol)
            #colony[k].set_fitness(fit)

            # # another approach: remove the worst sol in colony and add the current ant k
            # colony.sort(key=lambda x: x.fitness, reverse=False)  # ascending order
            # del colony[-1:]
            # colony.append(IndividualSolution())
            # colony[-1].set_solution(sol)
            # colony[-1].set_fitness(fit)
            colony.sort(key=lambda x: x.fitness, reverse=False)  # ascending order
            del colony[-1:]
            colony.append(IndividualSolution())
            colony[-1].set_solution(sol)
            colony[-1].set_fitness(fit)
            colony[-1].set_vehidx(vidx)

        # update pheromone
        tau_itr = env.update_pheromone(colony)
        tau.append(copy.deepcopy(tau_itr))

        # update probability
        prob_itr = env.update_prob()
        prob.append(copy.deepcopy(prob_itr))

        # best solution by generation
        curr_min = min(colony, key=lambda x: x.fitness)
        minfit_by_itr.append(curr_min.fitness) # for plot
        if curr_min.fitness < best.fitness:
            best.set_solution(curr_min.solution)
            best.set_fitness(curr_min.fitness)
            best.set_vehidx(curr_min.vehidx)
        # print('min cost of #', t, ': ', curr_min.fitness, 'no of veh: ', 1+max(curr_min.vehidx))

    # depot is added at first and last in a best list
    pi_ = get_clean_path(best.solution)
    # sr_ = copy.deepcopy(best.service)
    # routes = get_routes_from_list(pi_)

    # sr_ = copy.deepcopy(best.service)
    vehidx_ = copy.deepcopy(best.vehidx)
    # vehtype_ = copy.deepcopy(best.vehtype)

    if pi_[0] != 0: # if the first is not depot
        pi_.insert(0,0) # add depot at first
        vehidx_.insert(0,vehidx_[0])
        # vehtype_.insert(0,vehtype_[0])
    if pi_[-1] != 0: # if the last is not depot
        pi_.append(0) # add depot at last
        vehidx_.append(vehidx_[-1])
        # vehtype_.append(vehtype_[-1])
    # routes, routes_fit, serv_range, routes_vidx, routes_vtype = get_mt_hf_routes_from_list(pi_, xy, vehidx_, sr_)
    cost, vehicles = get_my_mt_hf_routes_from_list(matrix, pi_)
    # print('---------')
    # print('Total cost: ', cost)
    # print('Total Vehicles: ', vehicles)
    # print('Routes: ', pi_)
    # print('Service range: ', serv_range)
    # print('Vehicle index: ', routes_vidx)
    # print('Vehicle type: ', routes_vtype)
    return cost, (time() - t1)
    # print(f'run time:{time() - t1}s')

    # print('---------')
    # print('Total cost: ', best.fitness, ', Routes: ', routes)

    # plot the aco history and solution
    # plt.figure()
    # plt.plot(minfit_by_itr)

    # plot_route(loc, routes, best.fitness)


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
    try:
        matrix = data['tdtt2'].cpu().numpy()  # np (n_customer,n_customer, n_time_period)
    except:
        matrix = data['tdtt'].cpu().numpy()  # np (n_customer,n_customer, n_time_period)
    # print("matrix loaded")

    n_customer = len(customer_xy)
    # print("n_customer loaded")

    CAPACITIES = 1 #CAPACITIES[n_customer]

    # dist = get_dist_mat(xy) # distance matrix including depot

    return ACO(xy, matrix, demands, CAPACITIES)


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
    out_file = 'log_ACO_VRP' + str(size) + '-' + str(time()) + '.txt'
    if size == 10:
        NUM_iterations = 1200
    elif size == 20:
        NUM_iterations = 2000
    elif size == 50:
        NUM_iterations = 5000
    elif size == 100:
        NUM_iterations = 10000
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
                            "Total Iterations:" + str(iteration), 
                            "\n Final Total Cost:" + str(result), 
                            "\n Final Total Time:" + str(run_time), 
                            "\n Avg Cost:" + str(result / iteration), 
                            "\n Avg Time:" + str(run_time / iteration), 
                            "\n ----- \n \n"])
        handle.close()


   
