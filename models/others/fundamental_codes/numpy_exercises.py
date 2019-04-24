import numpy as np
#1
print(np.__version__)
np.show_config()
#2
#Create a null vector of size 10
Z = np.zeros(10)
print(Z)

# how to find the memory size of any array
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))

#
Z = np.zeros(10)
Z[4] = 1
print(Z)

# 7. Create a vector with values ranging from 10 to 49
Z = np.arange(10, 50)
print(Z)

# reverse
Z = np.arange(50)
Z = Z[::1]
print(Z)

# create a 3*3 matrix
Z = np.arange(9).reshape(3, 3)
print(Z)

# find indices of non-zero
nz = np.nonzero([1,2,3,0,1,4,0])



#100 Compute bootstrapped 95% confidence intervals for the mean of a 1D array X
# (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means).

# Author: Jessica B. Hamrick

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)