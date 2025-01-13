from radgrad.radgrad import exp, grad


def sigm(x):
    return 1 / (1 + exp(-x))


print(sigm(0.5))

fg = grad(sigm)
print(fg(0.5))
