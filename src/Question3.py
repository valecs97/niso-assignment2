import copy
import math
import time

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

from src.Question13 import evaluate2
from src.Question2 import exprFitness, exprFitness2, readData

one = ['data', 'sqrt', 'log', 'exp']
two = ['add', 'sub', 'mul', 'div', 'pow', 'max', 'diff', 'avg']
four = ['ifleq']


class Expression:
    def __init__(self, expr, x, y=None, z1=None, z2=None):
        self.expr = expr
        self.x = x
        self.y = y
        self.z1 = z1
        self.z2 = z2

    def evaluate(self, inp, n):
        return evaluate2(self.toExpr(), inp, n)

    def toExpr(self):
        if self.expr == 'nothing':
            return self.x
        elif self.expr == 'data':
            return [self.expr, self.x]
        elif self.expr == 'sqrt' or self.expr == 'log' or self.expr == 'exp':
            return [self.expr, self.x.toExpr()]
        elif self.expr == 'ifleq':
            return [self.expr, self.x.toExpr(), self.y.toExpr(), self.z1.toExpr(), self.z2.toExpr()]
        else:
            return [self.expr, self.x.toExpr(), self.y.toExpr()]

    def __str__(self):
        if self.expr == 'nothing':
            return str(self.x)
        elif self.expr == 'data':
            return '(' + self.expr + ' ' + str(self.x) + ')'
        elif self.expr == 'sqrt' or self.expr == 'log' or self.expr == 'exp':
            return '(' + self.expr + ' ' + str(self.x) + ')'
        elif self.expr == 'ifleq':
            return '(' + self.expr + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.z1) + ' ' + str(self.z2)
        else:
            return '(' + self.expr + ' ' + str(self.x) + ' ' + str(self.y) + ')'

    def findExpr(self, chromosomes):
        current = self
        while True:
            if len(current) <= chromosomes + 1:
                return current
            if current.x is not None and len(current.x) >= chromosomes:
                current = current.x
            elif current.y is not None and len(current.y) >= chromosomes:
                current = current.y
            elif current.z1 is not None and len(current.z1) >= chromosomes:
                current = current.z1
            elif current.z2 is not None and len(current.z2) >= chromosomes:
                current = current.z2
            else:
                return current

    def __len__(self):
        if self.z1 is not None and self.z2 is not None:
            return len(self.x) + len(self.y) + len(self.z1) + len(self.z2) + 1
        elif self.y is not None:
            return len(self.x) + len(self.y) + 1
        elif isinstance(self.x, Expression):
            return len(self.x) + 1
        else:
            return 0

    def __getitem__(self, item):
        try:
            if item == 0:
                return self
            elif item - 1 <= len(self.x):
                return self.x[item - 1]
            elif item - len(self.x) - 1 <= len(self.y):
                return self.y[item - len(self.x) - 1]
            elif item - len(self.x) - len(self.y) - 1 <= len(self.z1):
                return self.z1[item - len(self.x) - len(self.y) - 1]
            elif item - len(self.x) - len(self.y) - len(self.z1) - 1 <= len(self.z2):
                return self.z2[item - len(self.x) - len(self.y) - len(self.z1) - 1]
            else:
                return None
        except TypeError:
            return self

    def __setitem__(self, key, value):
        expr = self.__getitem__(key)
        if isinstance(value, Expression) and expr is not None:
            expr.expr = value.expr
            expr.x = value.x
            expr.y = value.y
            expr.z1 = value.z1
            expr.z2 = value.z2

    def __add__(self, other):
        peak = self.duplicate()
        current = peak
        prev = None
        while current.expr != 'data' and current.expr != 'nothing':
            prev = current
            if bool(random.getrandbits(1)):
                current = current.x
            elif current.y is not None:
                current = current.y
            else:
                current = current.x
        if prev is None:
            return other.duplicate()
        else:
            if prev.x == current:
                prev.x = other.duplicate()
            else:
                prev.y = other.duplicate()
            return peak

    def duplicate(self):
        if self.expr == 'nothing':
            return Expression('nothing', self.x)
        elif self.expr == 'data':
            return Expression(self.expr, self.x)
        elif self.expr == 'sqrt' or self.expr == 'log' or self.expr == 'exp':
            return Expression(self.expr, self.x.duplicate())
        elif self.expr == 'ifleq':
            return Expression(self.expr, self.x.duplicate(), self.y.duplicate(), self.z1.duplicate(),
                              self.z2.duplicate())
        else:
            return Expression(self.expr, self.x.duplicate(), self.y.duplicate())


class Fitness:
    def __init__(self, expr):
        self.expr = expr.toExpr()
        self.fitness = 0.0

    def evaluateFitness(self, x, y, n, m):
        self.fitness = exprFitness2(self.expr, n, m, x, y)
        return self.fitness


class GA:
    def __init__(self, x, y, n, m, minChro, maxChro):
        self.x = x
        self.y = y
        self.n = n
        self.m = m
        self.minChro = minChro
        self.maxChro = maxChro

        self.simpleExpr = ['data', 'sqrt', 'log', 'exp', 'add', 'sub', 'mul', 'div', 'pow', 'max', 'diff', 'avg',
                           'ifleq']

    def createExpr(self, chromosomes):
        chromosomes -= 1
        choice = 0
        if chromosomes >= 4:
            choice = random.randint(1, 12)
        elif chromosomes >= 2:
            choice = random.randint(1, 11)
        elif chromosomes >= 1:
            choice = random.randint(1, 3)

        if choice == 0:
            if random.randint(1, 100) < 70:
                return Expression('data', random.randint(0, self.n - 1))
            else:
                return Expression('nothing', random.random() * 10)
        elif choice == 12:
            chromoX = random.randint(2, chromosomes - 2)
            chromoY = chromosomes - chromoX

            aux = chromoX
            chromoX = random.randint(1, chromoX - 1)
            chromoXZ = aux - chromoX

            aux = chromoY
            chromoY = random.randint(1, chromoY - 1)
            chromoYZ = aux - chromoY

            return Expression('ifleq', self.createExpr(chromoX), self.createExpr(chromoY), self.createExpr(chromoXZ),
                              self.createExpr(chromoYZ))
        elif choice >= 4:
            chromoX = random.randint(1, chromosomes - 1)
            chromoY = chromosomes - chromoX
            return Expression(self.simpleExpr[choice], self.createExpr(chromoX), self.createExpr(chromoY))
        else:
            return Expression(self.simpleExpr[choice], self.createExpr(chromosomes))

    def initialPopulation(self, popSize):
        population = []
        for i in range(0, popSize):
            chromosomes = random.randint(self.minChro, self.maxChro)
            population.append(self.createExpr(chromosomes))
        return population

    def rankRoutes(self, population):
        fitnessResults = {}
        for i in range(0, len(population)):
            if Fitness(population[i]).evaluateFitness(self.x, self.y, self.n, self.m) != 0:
                fitnessResults[i] = 1 / Fitness(population[i]).evaluateFitness(self.x, self.y, self.n, self.m)
            else:
                fitnessResults[i] = 0
        return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

    def selection(self, popRanked, eliteSize):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(0, eliteSize):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - eliteSize):
            pick = 100 * random.random()
            for j in range(0, len(popRanked)):
                if pick <= df.iat[j, 3]:
                    selectionResults.append(popRanked[j][0])
                    break
        return selectionResults

    def matingPool(self, population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    def breed(self, parent1, parent2):
        childChromo = random.randint(min(len(parent1), len(parent2)), max(len(parent1), len(parent2)))
        geneA = min(childChromo - 1, int(random.random() * len(parent1)))
        geneB = childChromo - geneA

        childP1 = parent1.findExpr(geneA)
        childP2 = parent2.findExpr(geneB)

        child = childP1 + childP2

        return child

    def breedPopulation(self, matingpool, eliteSize):
        children = []
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            child = self.breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def mutate(self, individual, mutationRate):
        for swapped in range(len(individual)):
            if (random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))

                if isinstance(individual[swapped], Expression):
                    expr1 = individual[swapped].duplicate()
                else:
                    expr1 = individual[swapped]
                if isinstance(individual[swapWith], Expression):
                    expr2 = individual[swapWith].duplicate()
                else:
                    expr2 = individual[swapWith]

                individual[swapped] = expr2
                individual[swapWith] = expr1

        return individual

    def mutatePopulation(self, population, mutationRate):
        mutatedPop = []

        for ind in range(0, len(population)):
            mutatedInd = self.mutate(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        return mutatedPop

    def nextGeneration(self, currentGen, eliteSize, mutationRate):
        popRanked = self.rankRoutes(currentGen)
        selectionResults = self.selection(popRanked, eliteSize)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool, eliteSize)
        nextGeneration = self.mutatePopulation(children, mutationRate)
        return nextGeneration

    def geneticAlgorithmPlot(self, popSize, eliteSize, mutationRate, seconds):
        start = time.time()
        time.clock()

        pop = self.initialPopulation(popSize)
        caca = self.rankRoutes(pop)

        minim = 1 / caca[0][1]
        expr = str(pop[caca[0][0]])
        gen = -1
        i = 0

        while time.time() - start < seconds:
            pop = self.nextGeneration(pop, eliteSize, mutationRate)
            caca = self.rankRoutes(pop)
            prog = 1 / caca[0][1]
            if prog < minim:
                minim = prog
                expr = str(pop[caca[0][0]])
                gen = i
                # print(str(i) + " " + str(minim))
            i += 1
        # print("Best fitness: " + str(minim))
        # print("Best expression: " + str(expr))
        # print("Generation: " + str(gen))
        # print("Number of generations: " + str(i))
        # print("Time elapsed: " + str(time.time() - start))
        print(expr)
        return minim


# if __name__ == '__main__':
#     x, y = readData('apple.csv')
#     results = {}
#     absoluteMinim = {}
#     ga = GA(x, y, len(x[0]), len(y), 3, 10)
#     for run in range(100):
#         print(run)
#         for i in range(1):
#             # print(i)
#             res = ga.geneticAlgorithmPlot(popSize=75, eliteSize=35, mutationRate=0.01, seconds=30)
#             if i not in results:
#                 results[i] = 0
#             if i not in absoluteMinim:
#                 absoluteMinim[i] = 9999999
#             results[i] += res
#             if res < absoluteMinim[i]:
#                 absoluteMinim[i] = res
#     for i in range(1):
#         print(str(i) + " " + str(results[i] / 100))
#     for i in range(1):
#         print("Absolute minim: " + str(i) + " " + str(absoluteMinim[i]))
