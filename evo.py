from typing import List
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product
import math

class Population:
    def __init__(self,size):
        population_generator = product(np.random.randint(0, 100, size=size),repeat=1)
        self.pop = []
        for prod in population_generator:
            self.pop.append(prod)


def eval_fitness(pop, max_kills=300):
    # f(x,y) = 5*cos(x*y)
    to_delete = []
    fitness = []
    for i in range(len(pop)):
        print(len(pop))
        import pdb; pdb.set_trace()
        fitness.append(5*math.cos(pop[i][0]*pop[i][1])) # FUNCTION
    for i, val in enumerate(fitness):
        if i >= max_kills:
            break
        if val < 4.99 or val > 5: # CONDITION
            to_delete.append(i)
    pop = np.delete(pop, to_delete)
    return pop, len(to_delete)


def pairs(pop):
    pairs = np.array_split(pop, 2)
    for p1, p2 in zip(pairs[0], pairs[1]):
        yield p1, p2


def reproduce(pop, num_to_create):
    pair_iter = pairs(pop)
    children = []
    for _ in range(num_to_create):
        children.append(np.mean(next(pair_iter)))
    return mutate(np.append(pop, children))


def mutate(pop, num_to_mutate=40):
    idx = np.random.randint(0, len(pop), num_to_mutate)
    for i in idx:
        pop[i] += random.randint(-100, 100)
    return pop


def evolve(pop):
    pop, num_killed = eval_fitness(pop)
    return reproduce(pop, num_killed)


if __name__ == "__main__":
    num_epochs = 500
    population = Population(1000)
    x_vals = [population.pop[i][0] for i in range(len(population.pop))]
    y_vals = [population.pop[i][1] for i in range(len(population.pop))]

    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    print(x_mean,y_mean)

    x_means = []
    y_means = []
    pop_means = [(x_means,y_means)]
    for _ in range(num_epochs):
        pop = evolve(population.pop)
        print(x_mean,y_mean)
        x_means.append(x_mean)
        y_means.append(y_mean)
        population.pop = pop
    plt.plot(range(num_epochs), x_means,'g', range(num_epochs), y_means, 'b')
    plt.show()
