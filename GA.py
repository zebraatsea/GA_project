#  genetic algorithm cs 461 assignment
#  brooklyn dressel

import math
import random


# define function to maximize
def f(x, y):
    return (pow(1 - x, 2) * pow(math.e, -(x * x) - pow(y + 1, 2)) -
            (x - pow(x, 3) - pow(y, 3)) * pow(math.e, -(x * x) - (y * y)))


# initialize random population
def initialize(size, xy_range):
    return [(random.uniform(*xy_range), random.uniform(*xy_range)) for _ in range(size)]


# fitness eval
def fitness_eval(pop):
    return [f(indiv[0], indiv[1]) for indiv in pop]


# parent selection
def select_parents(pop, fit_lvl, k=3):
    parents = []
    for _ in range(len(pop)):
        candidates = random.sample(list(zip(pop, fit_lvl)), k)
        parents.append(max(candidates, key=lambda indiv: indiv[1])[0])
    return parents


# crossover
def crossover(p1, p2, cross_prob):
    if random.random() < cross_prob:
        alpha = random.random()  # pick crossover point
        child1 = (alpha * p1[0] + (1 - alpha) * p2[0],
                  alpha * p1[1] + (1 - alpha) * p2[1])  # combine parents to make child
        return child1
    return p1  # if no crossover, return parent (child is clone of parent)


# mutation
def mutate(individual, mutate_prob, xy_range):
    if random.random() < mutate_prob:
        return random.uniform(*xy_range), random.uniform(*xy_range)
    return individual


# put it all together
def genetic_alg(size, no_gens, crossover_prob, mutation_prob, xy_range):
    pop = initialize(size, xy_range)
    for _ in range(no_gens):
        fit_lvl = fitness_eval(pop)
        parents = select_parents(pop, fit_lvl)
        offspring = [crossover(random.choice(parents), random.choice(parents),
                    crossover_prob) for _ in range(len(parents))]
        pop = [mutate(indiv, mutation_prob, xy_range) for indiv in offspring]

    fitness = fitness_eval(pop)
    best_idx = fitness.index(max(fitness))
    return pop[best_idx], max(fitness)


# define parameters
pop_size = 8
number_o_gens = 200
crossover_probability = 0.7
mutation_probability = 0.01
range_of_xy = (-2, 2)

# run
best_individual, best_fitness = genetic_alg(pop_size, number_o_gens, crossover_probability,
                                            mutation_probability, range_of_xy)
print("Best (x,y): ", best_individual)
print("Maximum f(x, y): ", best_fitness)

