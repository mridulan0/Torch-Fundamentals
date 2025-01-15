import torch

z = torch.zeros(5, 5)
print(z)

i = torch.ones((5,5), dtype=torch.int16)
print(i)

# generating random tensors of a particular shape from a random seed
torch.manual_seed(1729)
r1 = torch.rand(2,2)
print("A random tensor:")
print(r1)

r2 = torch.rand(2,2)
print("\nA different random tensor:")
print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2,2)
print("\nShould match r1:")
print(r3)

# adding tensors together because they have the same shape
ones = torch.ones(2,3)
print(ones)

twos = torch.ones(2,3) * 2
print(twos)

threes = ones + twos
print(threes)
print(threes.shape)

# mathematical operations that are available to use
r = (torch.rand(2,2) - 0.5) * 2
print(r)

print("\nAbsolute value of r:")
print(torch.abs(r))

print("\nInverse sine of r:")
print(torch.asin(r))

print("\nDeterminant of r:")
print(torch.det(r))
print("\nSingular value decomposition of r:")
print(torch.svd(r))

print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))