from collections import namedtuple
stock = namedtuple('Stock', ['name', 'shares', 'date', 'time'])
stock_prototype = stock('', 0, 0.0, None, None)

def dict2stock(s):
    return stock_prototype._replace(**s)

a = {'name': 'ACME', 'shares': 100, 'price': 123.45}
dict2stock(a)

c = {'y':2,'z':4}
len(c)
list(c.keys())
list(c.values())

line = 'asdf fjdk; afed, fjek,asdf, foo'
import re
re.split(r'[;,\s]\s*', line) # ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']


