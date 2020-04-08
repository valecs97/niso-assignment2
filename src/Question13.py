from math import sqrt, log, exp


def add(params, inp, n):
    return evaluate2(params[0], inp, n) + evaluate2(params[1], inp, n)


def sub(params, inp, n):
    return evaluate2(params[0], inp, n) - evaluate2(params[1], inp, n)


def mul(params, inp, n):
    return evaluate2(params[0], inp, n) * evaluate2(params[1], inp, n)


def div(params, inp, n):
    if evaluate2(params[1], inp, n) != 0:
        return evaluate2(params[0], inp, n) / evaluate2(params[1], inp, n)
    return 0


def poww(params, inp, n):
    # print('POWWWWW')
    #
    # a = round(float(evaluate2(params[0], inp, n)),0)
    # print(evaluate2(params[0], inp, n))
    # b = round(float(evaluate2(params[1], inp, n)),0)
    # print(evaluate2(params[1], inp, n))
    # print(-22 ** 2.584962500721156)
    # print(pow(a,b))
    # print('END POWWWW')
    return (round(evaluate2(params[0], inp, n), 0) % 1000) ** (round(evaluate2(params[1], inp, n), 0) % 10)


def sqrtt(params, inp, n):
    return sqrt(abs(evaluate2(params[0], inp, n)))


def logg(params, inp, n):
    if abs(evaluate2(params[0], inp, n)) > 0:
        return log(abs(evaluate2(params[0], inp, n)), 2)
    return 0


def expp(params, inp, n):
    return exp(evaluate2(params[0], inp, n) % 10)


def maxx(params, inp, n):
    return max(evaluate2(params[0], inp, n), evaluate2(params[1], inp, n))


def ifleq(params, inp, n):
    if evaluate2(params[0], inp, n) <= evaluate2(params[1], inp, n):
        return evaluate2(params[2], inp, n)
    return evaluate2(params[3], inp, n)


def data(params, inp, n):
    return inp[int(abs(evaluate2(params[0], inp, n))) % n]


def diff(params, inp, n):
    return sub([data([params[0]], inp, n), data([params[1]], inp, n)], inp, n)


def avg(params, inp, n):
    k = int(abs(evaluate2(params[0], inp, n))) % n
    l = int(abs(evaluate2(params[1], inp, n))) % n
    if k - l == 0:
        return 0
    sm = 0
    for t in range(min(k, l), max(k, l)):
        sm += data([t], inp, n)
    return (1 / abs(k - l)) / sm


def evaluate2(expr, inp, n):
    if isinstance(expr, int) or isinstance(expr, float):
        return expr
    command = expr[0]
    if command == 'add':
        return add(expr[1:], inp, n)
    elif command == 'sub':
        return sub(expr[1:], inp, n)
    elif command == 'mul':
        return mul(expr[1:], inp, n)
    elif command == 'div':
        return div(expr[1:], inp, n)
    elif command == 'pow':
        return poww(expr[1:], inp, n)
    elif command == 'sqrt':
        return sqrtt(expr[1:], inp, n)
    elif command == 'log':
        return logg(expr[1:], inp, n)
    elif command == 'exp':
        return expp(expr[1:], inp, n)
    elif command == 'max':
        return maxx(expr[1:], inp, n)
    elif command == 'ifleq':
        return ifleq(expr[1:], inp, n)
    elif command == 'data':
        return data(expr[1:], inp, n)
    elif command == 'diff':
        return diff(expr[1:], inp, n)
    elif command == 'avg':
        return avg(expr[1:], inp, n)
    else:
        return expr


def testovici(inp, n):
    print(evaluate2(['sub', 8, ['data', 2]], inp, n))
    print(evaluate2(['log', 6], inp, n))
    print(evaluate2(['pow', -22, 2.584962500721156], inp, n))
    # print(evaluate2(['pow', ['sub', 8, ['data', 2]], ['log', 6]], inp, n))
