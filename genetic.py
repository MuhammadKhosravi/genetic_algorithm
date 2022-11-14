from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import random, randint


class Genetic:

    """
    NOTE:
        - S is the set of members.
        - T is the target value.
        - Chromosomes are represented as an array of 0 and 1 with the same length as the set.
        (0 means the member is not included in the subset, 1 means the member is included in the subset)

        Feel free to add any other function you need.
    """

    def __init__(self):
        pass

    def generate_initial_population(self, n: int, k: int) -> np.ndarray:
        """
        Generate initial population: This function is used to generate the initial population.

        Inputs:
        - n: number of chromosomes in the population
        - k: number of genes in each chromosome

        It must generate a population of size n for a set of k members.

        Outputs:
        - initial population
        """
        initial_population = np.zeros((k, n))

        for i in range(k):
            random_chromosom = np.random.choice([0, 1], size=(n,))
            initial_population[i, :] = random_chromosom

        return initial_population

    def objective_function(self, chromosome: np.ndarray, S: np.ndarray) -> int:
        """
        Objective function: This function is used to calculate the sum of the chromosome.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members

        It must calculate the sum of the members included in the subset (i.e. sum of S[i]s where Chromosome[i] == 1).

        Outputs:
        - sum of the chromosome
        """
        sum_of_subset = 0
        for i in range(chromosome.shape[0]):
            if chromosome[i] == 1:
                sum_of_subset += S[i]
        return sum_of_subset

    def is_feasible(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> bool:
        """
        This function is used to check if the sum of the chromosome (objective function) is equal or less to the target value.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members
        - T: target value

        Outputs:
        - True (1) if the sum of the chromosome is equal or less to the target value, False (0) otherwise
        """
        return self.objective_function(chromosome, S) <= T

    def cost_function(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> int:
        """
        Cost function: This function is used to calculate the cost of the chromosome.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members
        - T: target value

        The cost is calculated in this way:
        - If the chromosome is feasible, the cost is equal to (target value - sum of the chromosome)
        - If the chromosome is not feasible, the cost is equal to the sum of the chromosome

        Outputs:
        - cost of the chromosome
        """
        s = self.objective_function(chromosome, S)
        if self.is_feasible(chromosome, S):
            return T - s
        else:
            return s

    def selection(self, population: np.ndarray, S: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selection: This function is used to select the best chromosome from the population.

        Inputs:
        - population: current population
        - S: set of members
        - T: target value

        It select the best chromosomes in this way:
        - It gets 4 random chromosomes from the population
        - It calculates the cost of each selected chromosome
        - It selects the chromosome with the lowest cost from the first two selected chromosomes
        - It selects the chromosome with the lowest cost from the last two selected chromosomes
        - It returns the selected chromosomes from two previous steps

        Outputs:
        - two best chromosomes with the lowest cost out of four selected chromosomes
        """
        random_chromosomes = population[np.random.randint(
            random_chromosomes.shape[0], size=4), :]
        lowest_cost1, lowest_cost2 = None, None
        lowest_cost1 = max(random_chromosomes[0], random_chromosomes[1])
        lowest_cost2 = max(random_chromosomes[2], random_chromosomes[3])
        return lowest_cost1, lowest_cost2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, S: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover: This function is used to create two new chromosomes from two parents.

        Inputs:
        - parent1: first parent chromosome
        - parent2: second parent chromosome


        It creates two new chromosomes in this way:
        - It gets a random number between 0 and 1
        - If the random number is less than the crossover probability, it performs the crossover, otherwise it returns the parents
        - Crossover steps:
        -   It gets a random number between 0 and the length of the parents
        -   It creates two new chromosomes by swapping the first part of the first parent with the first part of the second parent and vice versa
        -   It returns the two new chromosomes as children


        Outputs:
        - two children chromosomes
        """

        random_prob = random()
        if not (random_prob < prob):
            return parent1, parent2
        random_index = randint(0, parent1.shape[0] - 1)
        child1, child2 = np.concatenate([parent1[:random_index + 1], parent2[random_index, :]]), np.concatenate([
            parent2[:random_index + 1], parent1[random_index, :]])
        return child1, child2

    def mutation(self, child1: np.ndarray, child2: np.ndarray, prob: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutation: This function is used to mutate the child chromosomes.

        Inputs:
        - child1: first child chromosome
        - child2: second child chromosome
        - prob: mutation probability

        It mutates the child chromosomes in this way:
        - It gets a random number between 0 and 1
        - If the random number is less than the mutation probability, it performs the mutation, otherwise it returns the children
        - Mutation steps:
        -   It gets a random number between 0 and the length of the children
        -   It mutates the first child by swapping the value of the random index of the first child
        -   It mutates the second child by swapping the value of the random index of the second child
        -   It returns the two mutated children

        Outputs:
        - two mutated children chromosomes
        """
        random_prob = random()
        if not (random_prob < prob):
            return child1, child2
        random_index = randint(0, child1.shape[0] - 1)
        child1 = 0 if child1[random_index] == 1 else 1 
        child2 = 0 if child2[random_index] == 1 else 1 
        return child1, child2

    def run_algorithm(self, S: np.ndarray, T: int, crossover_probability: float = 0.5, mutation_probability: float = 0.1, population_size: int = 100, num_generations: int = 100):
        """
        Run algorithm: This function is used to run the genetic algorithm.

        Inputs:
        - S: array of integers
        - T: target value

        It runs the genetic algorithm in this way:
        - It generates the initial population
        - It iterates for the number of generations
        - For each generation, it makes a new empty population
        -   While the size of the new population is less than the initial population size do the following:
        -       It selects the best chromosomes(parents) from the population
        -       It performs the crossover on the best chromosomes
        -       It performs the mutation on the children chromosomes
        -       If the children chromosomes have a lower cost than the parents, add them to the new population, otherwise add the parents to the new population
        -   Update the best cost if the best chromosome in the population has a lower cost than the current best cost
        -   Update the best solution if the best chromosome in the population has a lower cost than the current best solution
        -   Append the current best cost and current best solution to the records list
        -   Update the population with the new population
        - Return the best cost, best solution and records


        Outputs:
        - best cost
        - best solution
        - records
        """

        # UPDATE THESE VARIABLES (best_cost, best_solution, records)
        best_cost = np.Inf
        best_solution = None
        records = []

        # YOUR CODE HERE


        for i in tqdm(range(num_generations)):

            # YOUR CODE HERE
            pass

            records.append({'iteration': i, 'best_cost': best_cost,
                           'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE

        return best_cost, best_solution, records