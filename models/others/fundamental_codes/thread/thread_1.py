
# Performs a large calculation (CPU bound)
def some_work(args):
    print('do sth')

def some_threads():
    while True:
        print('do sth (THREAD)')


# Initiaze the pool
# 使用进程池的方式实现全局锁？
if __name__ == '__main__':
    import multiprocessing
    '''
    这个通过使用一个技巧利用进程池解决了GIL的问题。 
    当一个线程想要执行CPU密集型工作时，会将任务发给进程池。 
    然后进程池会在另外一个进程中启动一个单独的Python解释器来工作。 
    当线程等待结果的时候会释放GIL。 
    并且，由于计算任务在单独解释器中执行，那么就不会受限于GIL了。 
    在一个多核系统上面，你会发现这个技术可以让你很好的利用多CPU的优势。
    '''
    pool = multiprocessing.Pool()