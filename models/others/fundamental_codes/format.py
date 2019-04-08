# -*- coding:utf-8 -*-
import reprlib
import pprint
import textwrap


asd = set('supercalifragilisticexpialidocious')

print(reprlib.repr(asd))

t = [[[['black', 'cyan'], 'white', ['green', 'red']], [['magenta', 'yellow'], 'blue']]]

pprint.pprint(t, width=30)

doc = "The wrap() method is just like fill() except that it returns" \
      "a list of strings instead of one big string with newlines to separate" \
      "the wrapped lines."

print(textwrap.fill(doc, width=30))


'''
The locale module accesses a database of culture specific data formats. 
'''
import locale
# local_locale = locale.setlocale(locale.LC_ALL, 'English_United States.1252')
conv = locale.localeconv()
x = 12451235.7
format_x = locale.format("%d", x, grouping=True)

print(format_x)


import time, os.path
import bisect


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo

def insort_right(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2 # 求商的整数部分
        if x < a[mid]: hi = mid
        else: lo = mid+1
    a.insert(lo, x)

scores = [(100, 'perl'), (200, 'tcl'), (400, 'lua'), (500, 'python')]
temp = scores
final_scores = insort_right(scores, (300, 'ruby'))
print(temp)
bisect.insort(scores, (350, 'ruby2'))
print(scores)


#堆
from heapq import heapify, heappop, heappush
data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
heapify(data)
print(data)
heappush(data, -7) # ??? todo
print(data)
heappop(data, -7)
print(data)
[heappop(data) for i in range(3)]
print(data)

from decimal import *

# its true
round(Decimal('0.70') * Decimal('1.05'), 2)
