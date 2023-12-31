import json
import random
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy
from deap import base, creator, tools, algorithms
from deap.base import Toolbox
from deap.tools import Logbook

LOW, UP = -1, 42
CONFIG_FILE_NAME = "config.json"


def fitness_func(_x: int) -> float:
    return -.1 * _x ** 2 + 4 * _x + 7


def load_config() -> Dict[str, float]:
    with open(CONFIG_FILE_NAME) as config_file:
        return json.load(config_file)


def evaluate(_individual: List[int]) -> Tuple[float]:
    x = _individual[0]
    return fitness_func(x),


def init_deap():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def init_toolbox(_mutation_rate: float) -> Toolbox:
    _toolbox = base.Toolbox()
    _toolbox.register("attr_generator", random.randint, LOW, UP)
    _toolbox.register("individual", tools.initRepeat, creator.Individual, _toolbox.attr_generator, 2)
    _toolbox.register("population", tools.initRepeat, list, _toolbox.individual)
    _toolbox.register("evaluate", evaluate)
    _toolbox.register("mate", tools.cxTwoPoint)
    _toolbox.register("mutate", tools.mutUniformInt, low=LOW, up=UP, indpb=_mutation_rate)
    _toolbox.register("select", tools.selRoulette)
    return _toolbox


def plot_fitness_func(low: int, high: int):
    x_vals = range(low, high)
    y_vals = [fitness_func(x) for x in x_vals]
    plt.plot(x_vals, y_vals, label="Fitness function: $-0.1x^2 + 4x + 7$")
    plt.title("Fitness function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_stats(_stats: Logbook):
    gen = [s["gen"] for s in _stats]
    avg = [s["avg"] for s in _stats]
    std = [s["std"] for s in _stats]
    min_fit = [s["min"] for s in _stats]
    max_fit = [s["max"] for s in _stats]

    plt.figure(figsize=(20, 10))
    plt.plot(gen, avg, label="Average fitness")
    plt.plot(gen, min_fit, label="Minimum fitness")
    plt.plot(gen, max_fit, label="Maximum fitness")
    plt.fill_between(
        gen,
        [avg[i] - std[i] for i in range(len(_stats))],
        [avg[i] + std[i] for i in range(len(_stats))],
        color="gray", alpha=.2, label="Std deviation"
    )
    plt.title("Population fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    config = load_config()
    num_generations, population_size, crossover_rate, mutation_rate = config.values()
    init_deap()
    toolbox = init_toolbox(mutation_rate)
    population: List[List[int]] = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    _, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=num_generations,
        stats=stats,
        verbose=True
    )

    plot_stats(logbook)
    plot_fitness_func(LOW, UP)
