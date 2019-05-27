

def singleton(cls):
    instances = {}
    def wrapper(*args, **kargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kargs)
        return instances[cls]
    return wrapper


@singleton
class Foo(object):
    pass

foo1 = Foo()
foo2 = Foo()
print(foo1 is foo2)