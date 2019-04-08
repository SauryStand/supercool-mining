

class B(Exception):
    pass

class C(B):
    pass

class D(C):
    pass

for cls in [B, C, D]:
    try:
        raise cls()
    except D:
        print("D")
    except C:
        print("C")
    except B:
        print("B")



def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print("division by 0")
    else:
        print("result is: ", result)
    finally:
        print("excuting finally clause")



'''
The problem with this code is that it leaves the file open for an indeterminate 
amount of time after this part of the code has finished executing. 
This is not an issue in simple scripts, but can be a problem for larger applications.
'''
# for line in open("/Users/voyager2511/Documents/source_codes/others/supercool-mining/dataset/data/ch8/enrondata.txt"):
#     print(line)
#
# # proper one
# with open("/Users/voyager2511/Documents/source_codes/others/supercool-mining/dataset/data/ch8/enrondata.txt") as f:
#     for line in f:
#         print(line)



class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)
print(x.i)


class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    '''reverse the data'''
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index -1
        return self.data[self.index]

    def reverse(data):
        for index in range(len(data)-1, -1, -1):
            yield data[index]


X = Reverse('golf')
#data = iter(X.data)
for char in X.data:
    print(char)



for i in X.reverse('asdw'):
    print(i)



from math import pi, sin
sine_table = {x: sin(x*pi/180) for x in range(0, 91)}








