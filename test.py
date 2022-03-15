def fun(x):
    return x**2


def fuck(a, func):
    for f in a:
        func(f)


fuck(range(5), fun)
