from math import sqrt, log, exp


def add(params, inp, n):
    return evaluate(params[0], inp, n) + evaluate(params[1], inp, n)


def sub(params, inp, n):
    return evaluate(params[0], inp, n) - evaluate(params[1], inp, n)


def mul(params, inp, n):
    return evaluate(params[0], inp, n) * evaluate(params[1], inp, n)


def div(params, inp, n):
    if evaluate(params[1], inp, n) != 0:
        return evaluate(params[0], inp, n) / evaluate(params[1], inp, n)
    return 0


def poww(params, inp, n):
    return evaluate(params[0], inp, n) ** evaluate(params[1], inp, n)


def sqrtt(params, inp, n):
    return sqrt(evaluate(params[0], inp, n))


def logg(params, inp, n):
    return log(evaluate(params[0], inp, n), 2)


def expp(params, inp, n):
    return exp(evaluate(params[0], inp, n))


def maxx(params, inp, n):
    return max(evaluate(params[0], inp, n), evaluate(params[1], inp, n))


def ifleq(params, inp, n):
    if evaluate(params[0], inp, n) <= evaluate(params[1], inp, n):
        return evaluate(params[2], inp)
    return evaluate(params[3], inp)


def data(params, inp, n):
    return inp[abs(evaluate(params[0], inp, n)) % n]


def diff(params, inp, n):
    return diff([data(params[:0], inp, n), data(params[1:], inp, n)], inp, n)


def avg(params, inp, n):
    k = abs(evaluate(params[0], inp, n)) % n
    l = abs(evaluate(params[1], inp, n)) % n
    sm = 0
    for t in range(min(k, l), max(k, l)):
        sm += data([t], inp, n)
    return (1 / abs(k - l)) / sm


def evaluate(expr, inp, n):
    if isinstance(expr, int):
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
