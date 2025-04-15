import torch

x = torch.tensor(
    [
        [10, 20, 11, 4, 8, 20, 19, 20],
        [20, 30, 31, 20, 9, 22, 15, 90],
        [5, 6, 31, 5, 9, 20, 15, 90],
    ]
)

print(x.shape)

a, b = torch.where(x == 20)
print(a)
print(b)
