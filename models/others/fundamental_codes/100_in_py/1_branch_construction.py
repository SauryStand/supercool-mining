#-*- coding: utf-8 -*-
import math

score = float(input('pls input'))
if score >= 1:
    print(score)

a, b, c, = 2, 5, 4

if a+b > c and a+c > b and b+c > a:
    p = (a+b+c) /2
    area = math.sqrt(p * (p-a) * (p-b) * (p-c))

"""
range(101)可以产生一个0到100的整数序列。
range(1, 100)可以产生一个1到99的整数序列。
range(1, 100, 2)可以产生一个1到99的奇数序列，其中的2是步长，即数值序列的增量。
"""
# 1-100 sum
sum = 0
for x in range(101):
    sum += x
print(sum)

# 用for循环实现1~100之间的偶数求和
sum = 0
for x in range(2, 101, 2):
    sum += x

sum = 0
for x in range(1, 101):
    if x % 2 == 0:
        sum += x

import random
num = random.randint(1, 101)
counter = 0
while True:
    counter += 1
    number = int(input('pls input'))
    if number < num:
        print('bigger')
    elif number > num:
        print('smaller')
    else:
        print('bingo')
        break

for i in range(1, 10):
    for j in range(1, i+1):
        print('%d * %d = %d' % (i, j, i*j))

# 输入一个正整数判断它是不是素数
num = int(input('pls input'))
end = int(math.sqrt(num))
flag = True
for x in range(2, end + 1):
    if num % x == 0:
        flag = False
        break
if flag and num != 1:
    print('%d is prime' % num)

# 最大公倍数common multiple and common divisor
x = int(input('x = '))
y = int(input('y = '))
if x > y:
    x, y = y , x
for factor in range(x, 0, -1):
    if x % factor == 0 and y % factor == 0:
        print('%d and %d max common divisor is' % (x, y, factor))
        print('%d and %d max common multiple is' % (x, y, x*y // factor))
        break
'''
*
**
***
****
*****
'''
row = int(input('pls input'))
for i in range(row):
    for _ in range(i + 1):
        print('x')
'''
    *
   **
  ***
 ****
*****
'''
for i in range(row):
    for j in range(row):
        if j < row - i - 1:
            print(' ')
        else:
            print('*')
'''
    *
   ***
  *****
 *******
*********
'''
for i in range(row):
    for _ in range(row - i - 1):
        print(' ')
    for _ in range(2 * i + 1):
        print('x')











