# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:12:23 2020

@author: youle
"""
import random

from deap import base
from deap import creator
from deap import tools

def printConfig(npop, ndim, maxgen, pc, pm):

    print("Population Size: " + str(npop))
    print("Number of Variables: " + str(ndim))
    print("Max Generation: " + str(maxgen))
    print("Prbability of Crossover: " + str(pc))
    print("probability of Mutation: " +str(pm))

def evaluateOneMax(individual):
    # Do some computation
    return sum(individual),

npop = 100
ndim = 50
maxgen = 1000

pc = 0.9
pm = 1.0

printConfig(npop, ndim, maxgen, pc, pm)

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, ndim)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 1/ndim)
toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("evaluate", evaluateOneMax)

pop = toolbox.population(n = npop)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(maxgen):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < pc:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < pm:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    if (g + 1) % 50 == 0:
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("-" * 30)
        print("  Gen: "+str(g+1))
        print("  Min: %.5s" % min(fits))
        print("  Max: %.5s" % max(fits))
        print("  Avg: %.5s" % mean)
        print("  Std: %.5s" % std)
        print("-" * 30 + "\n")