import getopt
import sys


def readArgs():
    args = sys.argv[1:]
    question = -1
    clause = []
    assignment = "-1"
    for i in range(0, len(args), 2):
        opt = args[i]
        arg = args[i + 1]
        if opt == '-h':
            print('Start.py -q <question> -c <clause> -a <assignemnt>')
            sys.exit()
        elif opt in ("-q", "-question"):
            question = arg
        elif opt in ("-c", "-clause"):
            clause = arg.split(' ')
            clause = [int(x) for x in clause]
        elif opt in ("-a", "-assignment"):
            assignment = arg
    if question == 1:
        return question, clause, assignment


if __name__ == '__main__':
    readArgs()
