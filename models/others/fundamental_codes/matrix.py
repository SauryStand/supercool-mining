
import math

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]

var = [[row[i] for row in matrix] for i in range(4)]

print(var)


transposed = []
for i in range(4):
    transposed.append([row[i] for row in matrix])

print(transposed)

'''
second way
'''
transposed = []
for i in range(4):
    transposed_row = []
    for row in matrix:
        transposed_row.append(row[i])
    transposed.append(transposed_row)

print(transposed)


print(list(zip(*matrix)))

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}

'orange' in basket


a = set('abracadabra')
b = set('alacazam')

print(a - b)
print(a | b)
print(a ^ b)
print(a & b)









