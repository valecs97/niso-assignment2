from sklearn.metrics import mean_squared_error

from src.Question1 import evaluate


def exprFitness(expr, n, m, x, y):
    y_pred = []
    for inp in x:
        y_pred.append(evaluate(expr, inp, n))
    return mean_squared_error(y, y_pred)


def readData(fil):
    x = []
    y = []
    with open(fil, 'r') as f:
        for line in f.readlines():
            x.append(list(map(int, line.split()[:-1])))
            y.append(int(line.split()[-1]))
    return x, y
