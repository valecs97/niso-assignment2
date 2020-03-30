import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
from scipy import spatial
import time
from sklearn.metrics import mean_absolute_error
import random
import math
import time

data = None
priorityList = None
priorityList_counts = None
genePool = []
topList = None
modifiedGen = None
nDay = None


def run():
    global data, priorityList, priorityList_counts, genePool, topList, modifiedGen, nDay
    data = pd.read_csv('monthlyBeer.csv')
    data.columns = ['Month', 'beerProduction']
    data = data.bfill()
    beerProduction_mean = data.beerProduction.mean()
    beerProduction_DiffMean = data.beerProduction.diff().mean()
    beerProduction_Diff_Var = (np.var(data.beerProduction.diff())) ** (1 / 2)
    nDay = 60

    genePool = createPopulation(2000, nDay, beerProduction_mean, beerProduction_DiffMean)

    for i in range(5):
        print(i)
        data_train = justice_data(data, data.beerProduction, nDay)
        selected_genes = train(data, data.beerProduction, genePool, nDay)

        topList = []
        value = 0
        i = 0
        j = 0
        while (i <= len(selected_genes)):
            try:
                if selected_genes[j][1] == i:
                    value = selected_genes[i][0]
                    j += 1
                else:
                    topList.append(value)
                    print(value)
                    i += 1
            except IndexError:
                break

        priorityList, priorityList_counts = np.unique(topList, return_counts=True)
        modifiedGen = []
        crossover(data.beerProduction, nDay)
        priorityList, priorityList_counts = np.unique(topList, return_counts=True)
        modifiedGen = []

        modification(data.beerProduction, nDay)
        selectedGeans = genePool[2000:len(genePool)]

        last_gene, error = select_the_max(data.beerProduction, nDay)

        last_eliminated_gene = finalValueModif(data.beerProduction, nDay)

        plt.plot(last_eliminated_gene, color='green')
        plt.plot(data['beerProduction'].values[data.shape[0] - nDay:data.shape[0]], color='red')
        plt.ylabel('simulation result of ratios')
        plt.show()

        target = data.beerProduction[data.shape[0] - nDay:data.shape[0]]
        val_1 = last_eliminated_gene[0:nDay]
        org_1 = mean_absolute_error(val_1, target)
        print(org_1)


def justice_data(dataFrame, Series, day_range):
    global nDay
    for i in range(dataFrame.shape[0] - day_range * 4):
        values_mean = Series[dataFrame.shape[0] - day_range * 2:dataFrame.shape[0]].values.mean()
        Series[i:i + nDay * 2] = Series[i:i + nDay * 2] + (values_mean - Series[i:i + nDay * 2].mean())
    return dataFrame


def createPopulation(count, day_range, mean, diffmean):
    Population = []
    for i in range(count):
        gen = []
        for j in range(day_range * 2):
            gen.append(random.randint(int(mean - diffmean) - 1, int(mean + diffmean) + 1))
        Population.append(gen)
    return Population


def train(dataFrame, Series, genePool, day_range):
    selected_genes = []

    for i in range(dataFrame.shape[0] - day_range * 2):
        values = Series[i:i + day_range * 2].values
        min_mae = mean_absolute_error(genePool[0], values)
        for gen in genePool:
            mae = mean_absolute_error(gen, values)
            if mae < min_mae:
                min_mae = mae
                selected_genes.append([genePool.index(gen), i])

    return selected_genes


def crossover(Series, day_range):
    global data, priorityList, priorityList_counts, genePool, topList, modifiedGen
    genePool = np.array(genePool)
    for i in range(data.shape[0] - day_range * 4, data.shape[0] - day_range * 2):
        priorityList, priorityList_counts = np.unique(topList, return_counts=True)
        values = Series[i:i + day_range * 2].values
        run = True
        batch_threshold = 20
        batch = 0
        while (run):
            if batch >= batch_threshold:
                run = False
            genePool_Selected = np.random.choice(priorityList, 4,
                                                 p=priorityList_counts / sum(priorityList_counts))
            oldGen_1_15 = np.random.choice(genePool[genePool_Selected[0] - 1], int(day_range / 2))
            oldGen_2_15 = np.random.choice(genePool[genePool_Selected[1] - 1], int(day_range / 2))
            oldGen_3_15 = np.random.choice(genePool[genePool_Selected[2] - 1], int(day_range / 2))
            oldGen_4_15 = np.random.choice(genePool[genePool_Selected[3] - 1], int(day_range / 2))
            modifiedGen = np.concatenate((oldGen_1_15, oldGen_2_15, oldGen_3_15, oldGen_4_15), axis=None)
            target = mean_absolute_error(modifiedGen, values)
            val_1 = genePool[genePool_Selected[0] - 1]
            val_2 = genePool[genePool_Selected[1] - 1]
            val_3 = genePool[genePool_Selected[2] - 1]
            val_4 = genePool[genePool_Selected[3] - 1]
            thr_1 = mean_absolute_error(val_1, values)
            thr_2 = mean_absolute_error(val_2, values)
            thr_3 = mean_absolute_error(val_3, values)
            thr_4 = mean_absolute_error(val_4, values)
            if target < thr_1 and target < thr_2 and target < thr_3 and target < thr_4:
                print("Completed")
                genePool = np.vstack((genePool, modifiedGen))
                topList.append(len(genePool))
                batch += 1


def mutation_gen(gen, Series):
    x = np.random.choice(gen, 10)
    for i in range(len(gen)):
        if gen[i] in x:
            gen[i] = np.random.choice(Series.values)

    return gen


def modification(Series, day_range):
    global data, priorityList, priorityList_counts, genePool, topList, modifiedGen
    mutated_chromosome = np.zeros(day_range * 2)
    nonMutated_chromosome = np.zeros(day_range * 2)
    genePool = np.array(genePool)
    for i in range(data.shape[0] - day_range * 4, data.shape[0] - day_range * 2):
        priorityList, priorityList_counts = np.unique(topList, return_counts=True)
        values = Series[i:i + day_range * 2].values
        run = True
        batch_threshold = 100
        batch = 0
        while (run):
            if batch >= batch_threshold:
                run = False
            genePool_Selected = np.random.choice(priorityList, 1,
                                                 p=priorityList_counts / sum(priorityList_counts))
            mutated_chromosome = mutation_gen(list(genePool[genePool_Selected[0] - 1]), Series)
            nonMutated_chromosome = genePool[genePool_Selected[0] - 1]
            thr_1 = mean_absolute_error(mutated_chromosome, values)
            org_1 = mean_absolute_error(nonMutated_chromosome, values)
            batch += 1
            if thr_1 < org_1:
                print("Completed")
                genePool = np.vstack((genePool[:, 0], mutated_chromosome))
                topList.append(len(genePool))


def select_the_max(Series, day_range):
    global selectedGeans
    selectedGeans = list(selectedGeans)
    min_mae = mean_absolute_error(selectedGeans[0][0:day_range], Series[data.shape[0] - day_range:data.shape[0]])
    lastGen = []
    for gen in selectedGeans:
        if mean_absolute_error(gen[0:day_range], Series[data.shape[0] - day_range:data.shape[0]]) < min_mae:
            min_mae = mean_absolute_error(gen[0:day_range], Series[data.shape[0] - day_range:data.shape[0]])
            lastGen = gen
    return lastGen, min_mae


def finalValueModif(Series, day_range):
    global last_gene
    Last_gen = np.zeros(day_range)
    for i in range(100000):
        mutated_chromosome = mutation_gen(list(last_gene), Series)
        target = Series[data.shape[0] - day_range:data.shape[0]]
        val_1 = last_gene[0:day_range]
        org_1 = mean_absolute_error(val_1, target)
        thr_1 = mean_absolute_error(mutated_chromosome[0:day_range], target)
        if thr_1 < org_1:
            if (mean_absolute_error(mutated_chromosome[0:day_range], target)) < (
                    mean_absolute_error(Last_gen[0:day_range], target)):
                Last_gen = mutated_chromosome
                print("Completed")

    return Last_gen


if __name__ == '__main__':
    run()
