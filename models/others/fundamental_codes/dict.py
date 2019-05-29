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



str1 = "k:1|k1:2|k2:3|k3:4"

def str2dict(str1):
    dict = {}
    for items in str1.split('|'):
        key, value = items.split(':')
        dict[key] = value
    return dict

#字典推导式
d = {k:int(v) for t in str1.split('|') for k, v in (t.split(':'), )}

print(d)


words = [
       'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
       'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
       'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
       'my', 'eyes', "you're", 'under'
]

from collections import Counter
word_counts = Counter(words)
top_three = word_counts.most_common(3)
# print(top_three)

morewords = ['why','are','you','not','looking','in','my','eyes']
for word in morewords:
    word_counts[word] += 1

print(word_counts['you'])




