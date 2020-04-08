import getopt
import sys

from src.Question1 import evaluate
from src.Parser import parse_sexp
from src.Question13 import testovici
from src.Question2 import exprFitness, readData
from src.Question3 import GA

question = -1
input_vector = []
dimension = -1
expr = ""
size = 0
fil = ""
time_budget = 0
popSize = 0


def readArgs():
    args = sys.argv[1:]
    global question, input_vector, dimension, expr, size, fil,time_budget, popSize
    for i in range(0, len(args), 2):
        opt = args[i]
        arg = args[i + 1]
        if opt == '-h':
            print(
                'Start.py -q <question> -x <input> -n <dimension> -m <size> -expr <expression> -data <file> -time_budget <time> -lambda <pop_size>')
            sys.exit()
        elif opt in ("-q", "-question"):
            question = int(arg)
        elif opt in ("-x", "-input"):
            clause = arg.split(' ')
            input_vector = [float(x) for x in clause]
        elif opt in ("-n", "-dimension"):
            dimension = int(arg)
        elif opt in ("-m", "-size"):
            size = int(arg)
        elif opt in ("-e", "-expr"):
            expr = arg
        elif opt in ("-d", "-data"):
            fil = arg
        elif opt in ("-t", "-time_budget"):
            time_budget = int(arg)
        elif opt in ("-l", "-lambda"):
            popSize = int(arg)


if __name__ == '__main__':
    pass

readArgs()
expr = parse_sexp(expr)

if question == 1:
    print(evaluate(expr, input_vector, dimension))
elif question == 2:
    x, y = readData(fil)
    res = exprFitness(expr, dimension, size, x, y)
    print(res)
elif question == 3:
    x, y = readData(fil)
    ga = GA(x, y, dimension, size, 3, 10)
    ga.geneticAlgorithmPlot(popSize=popSize, eliteSize=10, mutationRate=0.001, seconds=time_budget)
