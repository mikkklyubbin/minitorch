"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float):
    return a * b


def id(a: float):
    return a


def add(a: float, b: float):
    return a + b


def neg(a: float):
    return -a


def lt(a: float, b: float):
    return a < b


def eq(a: bool, b: bool):
    return a == b


def max(a: float, b: float):
    if a < b:
        return b
    return a


def is_close(a: float, b: float):
    return abs(a - b) < 1e-6


def sigmoid(a: float):
    if a > 0:
        return 1 / (1 + math.exp(-a))
    return math.exp(a) / (1 + math.exp(a))


def relu(a: float):
    return max(0, a)


def log(a: float):
    return math.log(a)


def exp(a: float):
    return math.exp(a)


def inv(a: float):
    return 1 / a


def log_back(a: float, b: float):
    return b / a


def inv_back(a: float, b: float):
    return -b / (a * a)


def relu_back(a: float, b: float):
    if a > 0:
        return b
    return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(a: Iterable, b: Callable):
    """Higher-order function that applies a given function to each element of an iterable"""
    result = []
    for el in a:
        result.append(b(el))
    return result


def zipWith(a: Iterable, b: Iterable, c: Callable):
    """Higher-order function that combines elements from two iterables using a given function"""
    result = []
    for i in range(len(a)):
        result.append(c(a[i], b[i]))
    return result


def reduce(a: Iterable, b: Callable):
    """Higher-order function that reduces an iterable to a single value using a given function"""
    res = None
    for el in a:
        if res is None:
            res = el
        else:
            res = b(res, el)
    return res


def negList(ls: Iterable[float]):
    """Negate all elements in a list using map"""
    return map(ls, neg)


def addLists(ls: Iterable[float], ls2: Iterable[float]):
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(ls, ls2, add)


def sum(ls: Iterable[float]):
    """Sum all elements in a list using reduce"""
    res = reduce(ls, add)
    if res is None:
        return 0
    return res


def prod(ls: Iterable[float]):
    """Calculate the product of all elements in a list using reduce"""
    res = reduce(ls, mul)
    if res is None:
        return 1
    return res
