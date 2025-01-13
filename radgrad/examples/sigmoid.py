import radgrad.numpy_wrapper as np
from radgrad import grad


def sigm(x):
    return 1 / (1 + np.exp(-x))


print(sigm(0.5))

fg = grad(sigm)
print(fg(0.5))


def f(x1, x2):
    return np.log(x1) + x1 * x2 - np.sin(x2)


print(f(2, 5))
fg = grad(f)
print(fg(2, 5))

if __name__ == "__main__":
    pass
    # def f(x):
    #     return sin(x) + x

    # print(f(2))
    # fg = grad(f)
    # print(fg(2))

    # def f(x1, x2):
    #     return log(x1) + x1 * x2 - sin(x2)

    # print(f(2, 5))
    # fg = grad(f)
    # print(fg(2, 5))

    # def sigm(x):
    #     return 1 / (1 + exp(-x))

    # print(sigm(0.5))

    # fg = grad(sigm)
    # print(fg(0.5))

    # def tanh(x):
    #     y = exp(-2.0 * x)
    #     return (1.0 - y) / (1.0 + y)

    # fg = grad(tanh)
    # print(fg(1.0))

    # xx = np.linspace(-7, 7, 10)
    # print(tanh(xx))
    # print(fg(xx))
