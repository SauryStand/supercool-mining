# -*- coding:utf-8 -*-

string1, string2, string3, string4 = 'aa', 'Trondheim', 'Hammer Dance', ''

not_null = string1 and string2 or string3 or string4

print(not_null)



yes_votes = 42572654
no_votes = 43132495

percentage = float(yes_votes) / float((yes_votes + no_votes))
print('{:-9} YES votes  {:2.2%}'.format(yes_votes, percentage))


for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x * x, x * x * x))



# for key in {'one':1, 'two':2}:
#     print(key)
#
# for line in open("myfile.txt"):
#     print(line)



'''
有时候，你可能只想解压一部分，丢弃其他的值。对于这种情况 Python 并没有提供特殊的语法。 
但是你可以使用任意变量名去占位，到时候丢掉这些变量就行了。
'''
data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
_, shares, price, _ = data
# 你必须保证你选用的那些占位变量名在其他地方没被使用到。



