import getopt
import sys

from src.Question1 import evaluate
from src.Parser import parse_sexp
from src.Question2 import exprFitness, readData

question = -1
input_vector = []
dimension = -1
expr = ""
size = 0
fil = ""


def readArgs():
    args = sys.argv[1:]
    global question, input_vector, dimension, expr, size, fil
    for i in range(0, len(args), 2):
        opt = args[i]
        arg = args[i + 1]
        if opt == '-h':
            print('Start.py -q <question> -x <input> -n <dimension> -m <size> -expr <expression> -data <file>')
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
